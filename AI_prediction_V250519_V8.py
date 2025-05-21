import requests
import pandas as pd
import numpy as np
import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold('error')
import tensorflow as tf
import sys
import logging
import urllib3
import concurrent.futures
import warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # SSL 경고 비활성화
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
from io import BytesIO
from tensorflow.keras import models
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Dense, Layer
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import time
from tensorflow.keras.losses import MeanSquaredError
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import joblib  # Scaler 로딩/저장용
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter, date2num, HourLocator

HEADLESS = ('--auto' in os.getenv('PYTHON_ARGS','')) or bool(os.getenv('CLOUD_ENV'))
if not HEADLESS:
    from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QLabel, QPushButton, QCheckBox,
                                   QTableWidget, QTableWidgetItem, QTextBrowser, QMessageBox, QFileDialog, QVBoxLayout,
                                   QProgressBar, QWidget, QHBoxLayout, QSizePolicy)
    from PySide6.QtCore import (QCoreApplication, QRect, QSize, QMetaObject, Qt, QThread, Signal, QTimer)
    from PySide6.QtGui import (QBrush, QColor, QFont, QPixmap, QMovie, QIcon)
else:
    pass

import ssl
ssl._create_default_https_context = ssl._create_unverified_context   #전역 SSL 컨텍스트 우회

HEADLESS = ('--auto' in os.getenv('PYTHON_ARGS','')) or bool(os.getenv('CLOUD_ENV'))
# 클라우드 배포 시에만 /tmp를 사용
if os.getenv('CLOUD_ENV'):
    BASE_DIR = '/tmp'  
else:
    BASE_DIR = os.getenv('WORKDIR', os.path.dirname(os.path.abspath(__file__)))

# 시크릿으로 주입된 서비스 계정 JSON 을 파일로 덤프
creds_json = os.getenv('SERVICE_CREDENTIALS_JSON')
if creds_json:
    creds_path = os.path.join(BASE_DIR, 'service_credentials.json')
    with open(creds_path, 'w', encoding='utf-8') as f:
        f.write(creds_json)

LOCAL_WORKING_FOLDER = os.path.join(BASE_DIR, "prediction_temp_files")
os.makedirs(LOCAL_WORKING_FOLDER, exist_ok=True)

def resource_path(rel_path):
    # service_credentials.json, models/, scalers/ 등 모두 이 BASE_DIR 하위에서 찾도록
    return os.path.join(BASE_DIR, rel_path)

# TF 자체 로거 레벨 올리기
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow.keras.saving_utils').setLevel(logging.ERROR)

# 파이썬 FutureWarning, retracing 경고 등 무시
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*tf.function retracing.*')

# Google Drive API 관련 모듈 추가
from google.oauth2 import service_account  # type: ignore
from googleapiclient.discovery import build  # type: ignore
from googleapiclient.http import MediaFileUpload  # type: ignore

# --- 로깅 설정 ---
root = logging.getLogger()
root.setLevel(logging.INFO)
for h in root.handlers[:]: root.removeHandler(h)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
root.addHandler(handler)

# --- 사용자 정의 MultiHeadAttention 레이어 ---
@register_keras_serializable()
class MultiHeadAttention(Layer):  # type: ignore
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.Wq_list = [Dense(self.key_dim, name=f'wq_{i}') for i in range(self.num_heads)]
        self.Wk_list = [Dense(self.key_dim, name=f'wk_{i}') for i in range(self.num_heads)]
        self.Wv_list = [Dense(self.key_dim, name=f'wv_{i}') for i in range(self.num_heads)]
        self.Wo = Dense(feature_dim, name='wo')
        super().build(input_shape)

    def call(self, inputs):
        query = key = value = inputs  # Self-Attention 가정

        heads = []
        for i in range(self.num_heads):
            Q_h = self.Wq_list[i](query)  # (batch_size, sequence_length, key_dim)
            K_h = self.Wk_list[i](key)  # (batch_size, sequence_length, key_dim)
            V_h = self.Wv_list[i](value)  # (batch_size, sequence_length, key_dim)

            # Scaled Dot-Product Attention
            attention_scores = tf.matmul(Q_h, K_h, transpose_b=True)  # (batch_size, sequence_length, sequence_length)
            dk = tf.cast(self.key_dim, tf.float32)
            scaled_attention_scores = attention_scores / tf.sqrt(dk)

            attention_weights = tf.nn.softmax(scaled_attention_scores,
                                              axis=-1)  # (batch_size, sequence_length, sequence_length)

            head_output = tf.matmul(attention_weights, V_h)  # (batch_size, sequence_length, key_dim)
            heads.append(head_output)

        # 여러 헤드의 결과 연결
        if not heads:  # 헤드가 없는 경우 (num_heads=0), 입력을 그대로 반환 (이론상 발생 안함)
            return inputs

        concatenated_heads = tf.concat(heads, axis=-1)  # (batch_size, sequence_length, num_heads * key_dim)
        # 최종 출력 레이어 통과
        multi_head_output = self.Wo(concatenated_heads)  # (batch_size, sequence_length, feature_dim)
        return multi_head_output

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config

# --- 데이터 다운로드 스레드 ---
class DownloadThread(QThread):
    download_completed = Signal(dict)  # 다운로드 완료 시 데이터(dict)와 함께 시그널 발생
    progress_updated = Signal(int)  # 진행률(int) 업데이트 시그널
    errorOccurred = Signal(str)  # 오류 발생 시 오류 메시지(str)와 함께 시그널 발생

    def __init__(self, api_key, observation_codes, observation2_codes, start_time, end_time, save_path, max_workers=2):
        super().__init__()
        self.api_key = api_key
        self.observation_codes = observation_codes
        self.observation2_codes = observation2_codes
        self.start_time = start_time
        self.end_time = end_time
        self.save_path = save_path
        self.max_workers = max_workers

    def run(self):
        all_data = {}
        tasks = [(code, False) for code in self.observation_codes] \
                + [(code, True) for code in self.observation2_codes]
        total_tasks = len(tasks)
        completed_tasks = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.fetch_code_data, code, is_rainfall)
                : (code, is_rainfall)
                for code, is_rainfall in tasks
            }

            for future in concurrent.futures.as_completed(futures):
                code, is_rainfall = futures[future]
                try:
                    result_data = future.result()
                    key = f"{'rf' if is_rainfall else 'wl'}_{code}"
                    all_data[key] = result_data
                except Exception as e:
                    logging.error(f"[{code}] 다운로드 실패: {e}", exc_info=True)
                    self.errorOccurred.emit(f"[{code}] 다운로드 실패: {e}")
                    all_data[f"rf_{code}" if is_rainfall else f"wl_{code}"] = []

                completed_tasks += 1
                progress = int((completed_tasks / total_tasks) * 100)
                self.progress_updated.emit(progress)

        self.download_completed.emit(all_data)

    def fetch_code_data(self, code, is_rainfall=False):
        all_entries = []
        for s, e in self.date_range(self.start_time, self.end_time):
            if is_rainfall:
                xml_data = self.fetch_data_2(code, s, e)
                if xml_data:
                    parsed_data = self.parse_xml2(xml_data)
                    all_entries.extend(parsed_data)
            else:
                xml_data = self.fetch_data(code, s, e)
                if xml_data:
                    parsed_data = self.parse_xml(xml_data)
                    all_entries.extend(parsed_data)
        return all_entries

    def fetch_data(self, observation_code, start_time, end_time):
        url = f"https://api.hrfco.go.kr/761B0170-5681-4544-91A3-CFB7FF7AEA2C/waterlevel/list/1H/{observation_code}/{start_time.strftime('%Y%m%d%H%M')}/{end_time.strftime('%Y%m%d%H%M')}.xml"
        try:
            response = requests.get(url, verify=False, timeout=150)
            response.raise_for_status()
        except requests.Timeout as e:
            logging.warning(f"타임아웃: {url}")
            raise RuntimeError(f"{observation_code} 요청 타임아웃") from e
        except requests.HTTPError as e:
            code = e.response.status_code
            logging.error(f"HTTP {code} 오류: {url}")
            raise RuntimeError(f"{observation_code} HTTP 오류({code})") from e
        except requests.RequestException as e:
            logging.error(f"네트워크 오류: {e}")
            raise RuntimeError(f"{observation_code} 네트워크 오류") from e
        return response.text

    def fetch_data_2(self, observation_code, start_time, end_time):
        url = f"https://api.hrfco.go.kr/761B0170-5681-4544-91A3-CFB7FF7AEA2C/rainfall/list/1H/{observation_code}/{start_time.strftime('%Y%m%d%H%M')}/{end_time.strftime('%Y%m%d%H%M')}.xml"
        try:
            response = requests.get(url, verify=False, timeout=150)
            response.raise_for_status()
            return response.text
        except requests.Timeout as e:
            logging.warning(f"타임아웃(강수량): {url}")
            raise RuntimeError(f"{observation_code} 강수량 요청 타임아웃") from e
        except requests.HTTPError as e:
            code = e.response.status_code
            logging.error(f"강수량 HTTP {code} 오류: {url}")
            raise RuntimeError(f"{observation_code} 강수량 HTTP 오류({code})") from e
        except requests.RequestException as e:
            logging.error(f"강수량 네트워크 오류: {e}")
            raise RuntimeError(f"{observation_code} 강수량 네트워크 오류") from e

    def date_range(self, start_date, end_date, delta=timedelta(days=10)):
        current_date = start_date
        while current_date < end_date:
            yield current_date, min(current_date + delta, end_date)
            current_date += delta

    def parse_xml(self, xml_data):
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as e:
            logging.error("XML 파싱 실패: %s", e)
            raise RuntimeError("서버에서 잘못된 XML을 받았습니다.") from e

        data, seen_times = [], set()
        for waterlevel in root.findall('.//Waterlevel'):
            ymdhm = waterlevel.find('ymdhm').text if waterlevel.find('ymdhm') is not None else 'N/A'
            if ymdhm in seen_times:
                continue
            seen_times.add(ymdhm)
            wl_text = waterlevel.find('wl').text if waterlevel.find('wl') is not None else ''
            try:
                wl = float(wl_text) if wl_text.strip() else None
            except ValueError:
                wl = None
            time_formatted = f"{ymdhm[:4]}-{ymdhm[4:6]}-{ymdhm[6:8]} {ymdhm[8:10]}:00"
            data.append([time_formatted, wl])
        return data

    def parse_xml2(self, xml_data):
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as e:
            logging.error("XML 파싱 실패(강수량): %s", e)
            raise RuntimeError("서버에서 잘못된 강수량 XML을 받았습니다.") from e

        data, seen_times = [], set()
        for rainfall in root.findall('.//Rainfall'):
            ymdhm = rainfall.find('ymdhm').text if rainfall.find('ymdhm') is not None else 'N/A'
            if ymdhm in seen_times:
                continue
            seen_times.add(ymdhm)
            rf_text = rainfall.find('rf').text if rainfall.find('rf') is not None else ''
            rf = float(rf_text) if rf_text.strip() and rf_text.replace('.', '', 1).isdigit() else None
            time_formatted = f"{ymdhm[:4]}-{ymdhm[4:6]}-{ymdhm[6:8]} {ymdhm[8:10]}:00"
            data.append([time_formatted, rf])
        return data

# --- 댐 데이터 처리 함수 ---
def download_and_process_dam_data(damcd, start_date, end_date, is_automated=False):
    formatted_start_date = start_date.strftime('%Y-%m-%d')
    formatted_end_date = end_date.strftime('%Y-%m-%d')
    titles = {
        '1001210': '&title=%EA%B4%91%EB%8F%99%EB%8C%90',
        '1003110': '&title=%EC%B6%A9%EC%A3%BC%EB%8C%90',
        '1003611': '&title=%EC%B6%A9%EC%A3%BC%EC%A1%B0%EC%A0%95%EC%A7%80',
        '1006110': '&title=%ED%9A%A1%EC%84%B1%EB%8C%90',
        '1007601': '_bo&title=%EA%B0%95%EC%B2%9C%EB%B3%B4',
        '1007602': '_bo&title=%EC%97%AC%EC%A3%BC%EB%B3%B4',
        '1007603': '_bo&title=%EC%9D%B4%ED%8F%AC%EB%B3%B4',
        '1009710': '&title=%ED%8F%89%ED%99%94%EC%9D%98%EB%8C%90',
        '1012110': '&title=%EC%86%8C%EC%96%91%EA%B0%95%EB%8C%90',
        '1019901': '&title=%EA%B2%BD%EC%9D%B8%EC%95%84%EB%9D%BC%EB%B1%83%EA%B8%B8',
        '1021701': '&title=%EA%B5%B0%EB%82%A8%EB%8C%90',
        '1022701': '&title=%ED%95%9C%ED%83%84%EA%B0%95%EB%8C%90',
        '1024801': '_bo&title=%EA%B7%A4%ED%98%84%EB%B3%B4',
        '1302210': '&title=%EB%8B%AC%EB%B0%A9%EB%8C%90',
        '2001110': '&title=%EC%95%88%EB%8F%99%EB%8C%90',
        '2001611': '&title=%EC%95%88%EB%8F%99%EC%A1%B0%EC%A0%95%EC%A7%80',
        '2002110': '&title=%EC%9E%84%ED%95%98%EB%8C%90',
        '2002111': '&title=%EC%84%B1%EB%8D%95%EB%8C%90',
        '2002610': '&title=%EC%9E%84%ED%95%98%EC%A1%B0%EC%A0%95%EC%A7%80',
        '2004101': '&title=%EC%98%81%EC%A3%BC%EB%8C%90',
        '2007601': '_bo&title=%EC%83%81%EC%A3%BC%EB%B3%B4',
        '2008101': '&title=%EA%B5%B0%EC%9C%84%EB%8C%90',
        '2009601': '_bo&title=%EB%82%99%EB%8B%A8%EB%B3%B4',
        '2009602': '_bo&title=%EA%B5%AC%EB%AF%B8%EB%B3%B4',
        '2010101': '&title=%EA%B9%80%EC%B2%9C%EB%B6%80%ED%95%AD%EB%8C%90',
        '2011601': '_bo&title=%EC%B9%A0%EA%B3%A1%EB%B3%B4',
        '2011602': '_bo&title=%EA%B0%95%EC%A0%95%EA%B3%A0%EB%A0%B9%EB%B3%B4',
        '2012101': '_bo&title=%EB%B3%B4%ED%98%84%EC%82%B0%EB%8C%90',
        '2012210': '&title=%EC%98%81%EC%B2%9C%EB%8C%90',
        '2014601': '_bo&title=%EB%8B%AC%EC%84%B1%EB%B3%B4',
        '2014602': '_bo&title=%ED%95%A9%EC%B2%9C%EC%B0%BD%EB%85%95%EB%B3%B4',
        '2015110': '&title=%ED%95%A9%EC%B2%9C%EB%8C%90',
        '2017601': '_bo&title=%EC%B0%BD%EB%85%95%ED%95%A8%EC%95%88%EB%B3%B4',
        '2018110': '&title=%EB%82%A8%EA%B0%95%EB%8C%90',
        '2018611': '&title=%ED%95%A9%EC%B2%9C%EC%A1%B0%EC%A0%95%EC%A7%80',
        '2021110': '&title=%EB%B0%80%EC%96%91%EB%8C%90',
        '2021210': '&title=%EC%9A%B4%EB%AC%B8%EB%8C%90',
        '2022510': '&title=%EB%82%99%EB%8F%99%EA%B0%95%ED%95%98%EA%B5%BF%EB%91%91',
        '2101210': '&title=%EC%95%88%EA%B3%84%EB%8C%90',
        '2201220': '&title=%EC%82%AC%EC%97%B0%EB%8C%90',
        '2201230': '&title=%EB%8C%80%EC%95%94%EB%8C%90',
        '2201231': '&title=%EB%8C%80%EA%B3%A1%EB%8C%90',
        '2301210': '&title=%EC%84%A0%EC%95%94%EB%8C%90',
        '2403201': '&title=%EA%B0%90%ED%8F%AC%EB%8C%90',
        '2503210': '&title=%EC%97%B0%EC%B4%88%EB%8C%90',
        '2503220': '&title=%EA%B5%AC%EC%B2%9C%EB%8C%90',
        '3001110': '&title=%EC%9A%A9%EB%8B%B4%EB%8C%90',
        '3008110': '&title=%EB%8C%80%EC%B2%AD%EB%8C%90',
        '3008611': '&title=%EB%8C%80%EC%B2%AD%EC%A1%B0%EC%A0%95%EC%A7%80',
        '3010601': '_bo&title=%EC%84%B8%EC%A2%85%EB%B3%B4',
        '3012601': '_bo&title=%EA%B3%B5%EC%A3%BC%EB%B3%B4',
        '3012602': '_bo&title=%EB%B0%B1%EC%A0%9C%EB%B3%B4',
        '3203110': '_bo&title=%EB%B3%B4%EB%A0%B9%EB%8C%90',
        '3303110': '&title=%EB%B6%80%EC%95%88%EB%8C%90',
        '4001110': '&title=%EC%84%AC%EC%A7%84%EA%B0%95%EB%8C%90',
        '4007110': '&title=%EC%A3%BC%EC%95%94%EB%8C%90',
        '4104610': '&title=%EC%A3%BC%EC%95%94%EC%A1%B0%EC%A0%88%EC%A7%80%EB%8C%90',
        '4105210': '&title=%EC%88%98%EC%96%B4%EB%8C%90',
        '4204612': '&title=%EC%A3%BC%EC%95%94%EC%97%AD%EC%A1%B0%EC%A0%95%EC%A7%80',
        '5001701': '&title=%EB%8B%B4%EC%96%91%ED%99%8D%EC%88%98%EC%A1%B0%EC%A0%88%EC%A7%80',
        '5002201': '&title=%ED%8F%89%EB%A6%BC%EB%8C%90',
        '5003701': '&title=%ED%99%94%EC%88%9C%ED%99%8D%EC%88%98%EC%A1%B0%EC%A0%88%EC%A7%80',
        '5004601': '_bo&title=%EC%8A%B9%EC%B4%8C%EB%B3%B4',
        '5004602': '_bo&title=%EC%A3%BD%EC%82%B0%EB%B3%B4',
        '5101110': '&title=%EC%9E%A5%ED%9D%A5%EB%8C%90'
    }
    title = titles.get(str(damcd), 'default_title')
    url = f"https://www.water.or.kr/kor/realtime/excel/excelDown.do?mode=getExcelDataHDetailExcel{title}%20:%20{formatted_start_date}%20~%20{formatted_end_date}&damCd={damcd}&startDate={formatted_start_date}&endDate={formatted_end_date}"
    try:
        response = requests.get(url, verify=False, timeout=150)
        response.raise_for_status()
        excel_data = BytesIO(response.content)
    except requests.exceptions.RequestException as req_ex:
        error_msg = f"댐 데이터({damcd}) 다운로드 실패: {req_ex}"
        logging.error(error_msg)
        if not is_automated:  # is_automated 플래그 확인
            QMessageBox.critical(None, "댐 데이터 오류", error_msg)
        return pd.DataFrame()

    dam_columns = {
        '4001110': ['일시', '댐수위\n(EL.m)', '저수량\n(백만㎥)', '저수율\n(%)',
                    '강우량\n(mm)', '유입량\n(㎥/s)', '총방류량\n(㎥/s)', '본  류_수문\n(㎥/s)',
                    '_보조여수로\n(㎥/s)', '_소수력\n(본류)', '_소수력 바이패스관\n(㎥/s)', '_하천유지용수\n(㎥/s)',
                    '동  진  강_칠보발전\n(㎥/s)', '_운암소수력\n(㎥/s)', '_운암수갱\n(㎥/s)', '자체유입\n(㎥/s)', '방수로수위\n(EL.m)'],
        '4007110': ['일시', '댐수위\n(EL.m)', '방수로수위\n(EL.m)', '저수량\n(백만㎥)', '저수율\n(%)',
                    '강우량\n(mm)', '유입량\n(㎥/s)', '자체유입\n(㎥/s)', '조절지댐 신구도수유입\n(㎥/s)', '총방류량\n(㎥/s)',
                    '광천1호기\n(㎥/s)', '광천2,3호기\n(㎥/s)', '수문\n(㎥/s)', 'OUTLET 방류량\n(㎥/s)', '신도수터널\n(본댐→조절지)\n(㎥/s)',
                    '구도수터널\n(본댐→조절지)\n(㎥/s)', '보조여수로\n(㎥/s)', '광주취수량-광주천\n(㎥/s)', '광주천\n(㎥/s)']
    }

    if damcd == '4001110':
        header_row1 = pd.read_excel(excel_data, header=None, skiprows=3, nrows=1).iloc[0]
        header_row2 = pd.read_excel(excel_data, header=None, skiprows=4, nrows=1).iloc[0]
        headers = [f"{h1}_{h2}" if pd.notna(h2) else h1 for h1, h2 in zip(header_row1, header_row2)]

        df = pd.read_excel(excel_data, skiprows=4, header=None)
        df.columns = dam_columns[damcd]
    else:
        df = pd.read_excel(excel_data, skiprows=3)
        if damcd in dam_columns:
            df.columns = dam_columns[damcd]
        else:
            QMessageBox.critical(None, "Error", f"No column definitions for dam code {damcd}")
            return pd.DataFrame()  # 빈 데이터프레임 반환

    current_year = datetime.now().year

    def adjust_datetime(x):
        if pd.notnull(x):
            x = x.replace('시', '').strip()
            try:
                date_part, time_part = x.split()

                # 시간이 한 자리 숫자인 경우 앞에 0을 추가
                if len(time_part) == 1:
                    time_part = f"0{time_part}"

                if len(time_part) == 2:
                    time_part += ":00"

                if time_part == "24:00":
                    dt = datetime.strptime(f"{current_year}-{date_part} 00:00", '%Y-%m-%d %H:%M')
                    return dt + timedelta(days=1)
                else:
                    return datetime.strptime(f"{current_year}-{date_part} {time_part}", '%Y-%m-%d %H:%M')
            except ValueError:
                return None
        else:
            return None

    df['일시'] = df['일시'].apply(adjust_datetime)
    df.sort_values(by='일시', inplace=True)
    columns_to_sum_dict = {
        '4001110': ['본  류_수문\n(㎥/s)', '_보조여수로\n(㎥/s)', '_소수력\n(본류)', '_소수력 바이패스관\n(㎥/s)', '_하천유지용수\n(㎥/s)'],
        '4007110': ['광천1호기\n(㎥/s)', '광천2,3호기\n(㎥/s)', '수문\n(㎥/s)']
    }
    columns_to_sum = columns_to_sum_dict.get(damcd, [])
    if columns_to_sum:
        df[damcd] = df[columns_to_sum].sum(axis=1)
        selected_columns = ['일시', damcd]
        final_df = df[selected_columns]
        return final_df
    return pd.DataFrame()

# --- 예측 모델 클래스 ---
class Predictor:

    def __init__(self):
        self.scaler_x = None  # 입력 데이터 스케일러
        self.scaler_y = None  # 타겟 데이터 스케일러

    def load_scalers(self, scaler_x_filename, scaler_y_filename):
        try:
            self.scaler_x = joblib.load(scaler_x_filename)
            self.scaler_y = joblib.load(scaler_y_filename)
        except FileNotFoundError as fnf_err:
            logging.error(f"스케일러 파일 로드 실패: {fnf_err}")
            raise  # 오류를 다시 발생시켜 호출부에서 처리하도록 함
        except Exception as e_scaler:
            logging.error(f"스케일러 로딩 중 일반 오류: {e_scaler}", exc_info=True)
            raise

    def create_sequences(self, data, past, future):
        X = []
        for i in range(len(data) - past - future + 1):
            X.append(data[i:(i + past)])
        return np.array(X)

    def fit_scalers(self, data, target):
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaler_x.fit(data)
        self.scaler_y.fit(target)

    def process_and_predict(self, model_file, model_variables, data_file, target, past, future, is_automated_run=False):
        selected_variables = model_variables[target]
        data = pd.read_excel(data_file)
        if not data_file:
            if not is_automated_run:  # 자동 실행이 아닐 때만 메시지 박스 표시
                QMessageBox.information(None, "알림", "데이터 파일이 선택되지 않았습니다.")
            return None
        if '일시' not in data.columns:
            if not is_automated_run:
                QMessageBox.critical(None, "오류", "'일시' 열이 데이터 파일에 없습니다.")
            return
        try:
            data['일시'] = pd.to_datetime(data['일시'])
        except Exception as e:
            if not is_automated_run:
                QMessageBox.critical(None, "오류", f"'일시' 열을 날짜 형식으로 변환하는 중 오류가 발생했습니다. : {e}")
            return
        data.columns = data.columns.astype(str)
        data1 = data[[var for var in selected_variables if var in data.columns]]
        numeric_data = data1.select_dtypes(include=[np.number])

        # 모델 이름 추출
        model_name = os.path.basename(model_file).split('.')[0]

        # 스케일러 로드
        scaler_x_path = resource_path(os.path.join('scalers', f'scaler_x_{model_name}.pkl'))
        scaler_y_path = resource_path(os.path.join('scalers', f'scaler_y_{model_name}.pkl'))

        if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
            QMessageBox.critical(None, "오류", f"스케일러 파일을 찾을 수 없습니다: {scaler_x_path}, {scaler_y_path}")
            return

        self.load_scalers(scaler_x_path, scaler_y_path)

        model = None
        try:
            model = models.load_model(model_file, custom_objects={'MultiHeadAttention': MultiHeadAttention})
        except Exception as e:
            QMessageBox.critical(None, "오류", f"모델 로드 중 오류가 발생했습니다: {e}")
            return

        if model is None:
            return

        # 데이터 정규화 및 시퀀스 생성
        scaled_data = self.scaler_x.transform(numeric_data)
        X = []
        prediction_times = []
        time_delta = data['일시'].diff().mode()[0]  # 시간 간격 (1시간)
        num_samples = len(scaled_data)

        for i in range(past, num_samples + 1):
            if i > num_samples:
                break
            X.append(scaled_data[i - past:i])
            prediction_time = data['일시'].iloc[i - 1] + future * time_delta
            prediction_times.append(prediction_time)

        X = np.array(X)

        # 예측 수행
        if len(X) == 0:
            QMessageBox.critical(None, "오류", "입력 데이터가 충분하지 않아 예측을 수행할 수 없습니다.")
            return None

        predictions = model.predict(X, verbose=0)
        predictions_rescaled = self.scaler_y.inverse_transform(predictions).flatten()
        predictions_rescaled = np.round(predictions_rescaled, 3)

        # 예측 결과를 데이터프레임으로 정리
        comparison = pd.DataFrame({'일시': prediction_times, f'예측 수위({future}시간)': predictions_rescaled})

        return comparison


# --- 메인 UI 클래스 ---
class Ui_KIHS(object):
    def __init__(self):
        self.local_working_folder = 'prediction_temp_files'
        self.google_api_scopes = ['https://www.googleapis.com/auth/drive']
        self.google_file_id = "16g4Btk17vNHSTPy-b40kxCESY38g0cD-"
        self.service_account_credential_file = resource_path("service_credentials.json")
        self.is_headless = False
    STATION_NAME_MAP = {
        "1002698": "영월군(팔괴교)", "1006690": "원주시(문막교)", "1007625": "여주시(남한강교)",
        "1014650": "홍천군(홍천교)", "1014680": "홍천군(반곡교)", "1016670": "광주시(서하교)",
        "1101630": "천안시(안성천교)", "3009693": "대전시(신구교)", "3010660": "세종시(명학리)",
        "3011665": "청주시(미호강교)", "3011685": "세종시(미호교)", "3302658": "정읍시(초강리)",
        "4004615": "순창군(유적교)", "4004660": "남원시(신덕리)", "4004690": "곡성군(금곡교)",
        "4006660": "곡성군(고달교)", "4008670": "곡성군(태안교)", "4009610": "구례군(구례교)",
        "4009630": "구례군(송정리)", "4009640": "광양시(남도대교)", "5001640": "광주광역시(용산교)",
        "5001645": "광주광역시(첨단대교)"
    }
    api_key = "761B0170-5681-4544-91A3-CFB7FF7AEA2C"
    observation_codes = ['1002685', '1002687', '1002698', '1005697', '1006670', '1006672', '1006680', '1006690',
                             '1007615', '1007620', '1007625', '1014630', '1014640', '1014650', '1014665', '1014680',
                             '1016607', '1016650', '1016660', '1016670', '1016695', '1018620', '1018625', '1018630',
                             '1018635', '1018638', '1018665', '1018669', '1018670', '1018675', '1101610', '1101620',
                             '1101630', '1101635', '3004637', '3004645', '3008670', '3008695', '3009665', '3009670',
                             '3009673', '3009675', '3009680', '3009693', '3009698', '3010620', '3010660', '3011625',
                             '3011630', '3011635', '3011641', '3011643', '3011645', '3011660', '3011665', '3011685',
                             '3011695', '3012602', '3301652', '3301654', '3301655', '3301657', '3301665', '3302643',
                             '3302645', '3302653', '3302658', '4002640', '4002690', '4003690', '4004615', '4004660',
                             '4004690', '4005670', '4005690', '4006660', '4008660', '4008670', '4009610', '4009630',
                             '4009640', '5001615', '5001620', '5001625', '5001627', '5001640', '5001645', '5001655',
                             '5001660']
    observation2_codes = ['10024060', '10024170', '10024200', '10024220', '10024260', '10054010', '10054020',
                              '10064020', '10064030', '10064050', '10064060', '10064070', '10064080', '10064120',
                              '10074030', '10074100', '10144050', '10144060', '10144070', '10144080', '10144165',
                              '10164010', '10164030', '10164050', '10164075', '10164080', '10184010', '10184110',
                              '10184120', '10184140', '10184190', '10184220', '10224070', '11014020', '11014050',
                              '11014080', '11014110', '11014120', '11014130', '30014010', '30014040', '30014080',
                              '30014140', '30014160', '30034010', '30034020', '30044030', '30084050', '30084070',
                              '30084080', '30094020', '30094040', '30094050', '30104010', '30114020', '30114030',
                              '30114040', '30114060', '30114070', '30114100', '33014070', '33014100', '33024070',
                              '33024080', '40014050', '40014060', '40014070', '40024020', '40024030', '40024050',
                              '40024060', '40034010', '40034020', '40034030', '40044010', '40044030', '40044040',
                              '40044060', '40044070', '40054020', '40054030', '40054040', '40064010', '40074050',
                              '40074060', '40074070', '40074080', '40074082', '40074140', '40074143', '40084010',
                              '40084020', '40094060', '40094070', '40094080', '40094110', '40094120', '40094150',
                              '40094160', '40094170', '50014020', '50014030', '50014050', '50014060', '50014070',
                              '50024020']
    local_working_folder = 'prediction_temp_files'
    google_file_id = '16g4Btk17vNHSTPy-b40kxCESY38g0D-'
    service_account_credential_file = 'service_credentials.json'

    # 모델별 과거 데이터 필요 시간 (시간 단위)
    PAST_HOURS_3H = {
        '1002698': 48,
        '1006690': 48,
        '1007625': 48,
        '1014650': 48,
        '1014680': 24,
        '1016670': 48,
        # '1018630': 48,
        # '1018638': 48,
        # '1018675': 48,
        '1101630': 24,
        # '3008670': 24,
        # '3009680': 24,
        '3009693': 48,
        '3010660': 48,
        # '3011645': 48,
        '3011665': 48,
        '3011685': 48,
        '3302658': 12,
        '4004615': 24,
        '4004660': 24,
        '4004690': 24,
        '4006660': 24,
        '4008670': 12,
        '4009610': 36,
        '4009630': 48,
        '4009640': 48,
        '5001640': 24,
        '5001645': 24
    }
    PAST_HOURS_6H = {
        '1002698': 48,
        '1006690': 48,
        '1007625': 48,
        '1014650': 48,
        '1014680': 24,
        '1016670': 48,
        # '1018630': 48,
        # '1018638': 48,
        # '1018675': 48,
        '1101630': 24,
        # '3008670': 24,
        # '3009680': 24,
        '3009693': 24,
        '3010660': 48,
        # '3011645': 48,
        '3011665': 48,
        '3011685': 48,
        '3302658': 12,
        '4004615': 24,
        '4004660': 24,
        '4004690': 24,
        '4006660': 24,
        '4008670': 12,
        '4009610': 36,
        '4009630': 48,
        '4009640': 48,
        '5001640': 24,
        '5001645': 24
    }
    # 모델별 입력 변수 목록 (관측소 코드: [입력 변수 코드 리스트])
    model_variables = {
        '1002698': ['10024060', '10024170', '10024200', '10024220', '10024260', '1002685', '1002687', '1002698'],
        '1006690': ['10064020', '10064030', '10064050', '10064060', '10064070', '10064080', '10064120', '1006670',
                    '1006672', '1006680', '1006690'],
        '1007625': ['10054010', '10054020', '10064030', '10064050', '10064060', '10074030', '10074100', '1005697',
                    '1006690', '1007615', '1007620', '1007625'],
        '1014650': ['10144050', '10144060', '10144070', '10144080', '10144165', '1014630', '1014640', '1014650'],
        '1014680': ['10144050', '10144060', '10144070', '10144080', '10144165', '1014630', '1014640', '1014650',
                    '1014665', '1014680'],
        '1016670': ['10164010', '10164030', '10164050', '10164075', '10164080', '1016607', '1016650', '1016660',
                    '1016695', '1016670'],
        # '1018630': ['10184110', '10184120', '10184190', '10184220', '10224070', '1018620', '1018625', '1018635', '1018630'],
        # '1018638': ['10184110', '10184120', '10184190', '10184220', '10224070', '1018620', '1018625', '1018630', '1018635', '1018638'],
        # '1018675': ['10184010', '10184110', '10184140', '10184190', '1018665', '1018669', '1018670', '1018675'],
        '1101630': ['11014020', '11014050', '11014080', '11014110', '11014120', '11014130', '1101610', '1101620',
                    '1101635', '1101630'],
        # '3008670': ['30014080', '30034010', '30034020', '30044030', '30084050', '3004637', '3004645', '3008670'],
        # '3009680': ['30084070', '30094020', '30094040', '30094050', '3009665', '3009670', '3009673', '3009675', '3009693', '3009680'],
        '3009693': ['30084070', '30094020', '30094040', '30094050', '3009665', '3009670', '3009673', '3009675',
                    '3009680', '3009698', '3009693'],
        '3010660': ['30084070', '30084080', '30094020', '30094040', '30094050', '30104010', '3008695', '3009698',
                    '3010620', '3012602', '3010660'],
        # '3011645': ['30114020', '30114040', '3011641', '3011643', '3011645'],
        '3011665': ['30104010', '30114020', '30114030', '30114060', '30114070', '30114100', '3011625', '3011630',
                    '3011635', '3011660', '3011685', '3011665'],
        '3011685': ['30104010', '30114020', '30114030', '30114060', '30114070', '30114100', '3011625', '3011630',
                    '3011635', '3011660', '3011665', '3011695', '3011685'],
        # '3301654': ['33014070', '3301652', '3301654'],
        # '3301665': ['33014070', '33014100', '3301654', '3301657', '3301655', '3301665'],
        # '3302645': ['33024080', '40014050', '3302643', '3302645'],
        '3302658': ['33024080', '40014050', '50024020', '3302645', '3302653', '3302658'],
        '4004615': ['4001110', '40014050', '40014060', '40014070', '40024020', '40024030', '40034010', '40034020',
                    '40034030', '40044040', '4002690', '4002640', '4003690', '4004615'],
        '4004660': ['4001110', '40014050', '40014060', '40014070', '40024020', '40024030', '40034010', '40034020',
                    '40034030', '40044010', '40044060', '40044070', '4004615', '4004660'],
        '4004690': ['4001110', '40014050', '40014060', '40014070', '40024020', '40024030', '40034010', '40034020',
                    '40034030', '40044010', '40044060', '40044070', '4004660', '4004615', '4004690'],
        '4006660': ['4001110', '40024020', '40024030', '40034010', '40034020', '40034030', '40044010', '40044060',
                    '40054020', '40054030', '40054040', '40064010', '4004690', '4004660', '4005670', '4005690',
                    '4006660'],
        '4008670': ['4007110', '40084010', '40084020', '4008660', '4008670'],
        '4009610': ['4001110', '4007110', '40024030', '40024050', '40034010', '40034020', '40044010', '40044030',
                    '40044040',
                    '40044060', '40044070', '40054020', '40054030', '40054040', '40064010', '40074080', '40084010',
                    '40084020', '40094060', '40094110', '4008670', '4006660', '4004690', '4004660', '4009610'],
        '4009630': ['4001110', '40024030', '40024050', '40034010', '40034020', '40044010', '40044030', '40044040',
                    '40044060', '40044070', '40054020', '40054030', '40054040', '40064010', '40074080', '40084010',
                    '40084020', '40094060', '40094110', '4008670', '4006660', '4004690', '4004660', '4009610',
                    '4009630'],
        '4009640': ['4001110', '40024030', '40024050', '40034010', '40034020', '40044010', '40044030', '40044040',
                    '40044060', '40044070', '40054020', '40054030', '40054040', '40064010', '40074080', '40084010',
                    '40084020', '40094060', '40094070', '40094080', '40094110', '40094120', '40094150', '40094160',
                    '40094170', '4008670', '4006660', '4004690', '4004660', '4009610', '4009630', '4009640'],
        '5001640': ['50014030', '50014050', '50014060', '50014070', '5001615', '5001620', '5001625', '5001627',
                    '5001640'],
        '5001645': ['50014030', '50014050', '50014060', '50014070', '5001615', '5001620', '5001625', '5001627',
                    '5001640', '5001645'],
        # '5001660': ['50014020', '50014030', '50014050', '50014060', '50014070', '5001615', '5001620', '5001625',
        #            '5001627', '5001640', '5001645', '5001655', '5001660'],
    }

    def setupUi(self, KIHS=None):
        # ─── HEADLESS MODE INITIALIZATION ───
        # KIHS==None 이면 GUI 생성 대신 자동주기 모드에 필요한 최소 속성만 초기화
        if KIHS is None:
            # 1) 자동토글 체크박스 대체 객체
            class DummyCheckBox:
                def isChecked(self_inner): return True
                def setChecked(self_inner, v): pass
            self.checkBox_auto = DummyCheckBox()
            # 2) 더미 프로그레스바 (download_clicked 에서 setValue 호출 방어)
            class DummyProgressBar:
                def setValue(self, v): pass
            self.progressBar = DummyProgressBar()
            # 3) 더미 콤보박스: 전체 관측소 코드 리스트를 순환할 수 있도록
            class DummyComboBox:
                def __init__(self, items):
                    self._items = items
                def count(self):
                    return len(self._items)
                def itemText(self, i):
                    return self._items[i]
            station_codes = list(self.STATION_NAME_MAP.keys())
            self.comboBox = DummyComboBox(station_codes)
            # 4) 더미 타이머: isActive() 호출 방어
            class DummyTimer:
                def isActive(self): return False
                def start(self, ms=None): pass
                def stop(self): pass
            self._auto_timer = DummyTimer()
            # 5) 기타 GUI 위젯 더미
            self._download_has_errors = False
            self.textBrowser = None
            self.tableWidget = None
            self.tableWidget_2 = None
            return

        # --- Google Drive 및 로컬 경로 설정 ---
        # 스크립트 실행 위치 기준으로 'prediction_temp_files' 폴더 경로 설정
        # PyInstaller로 패키징된 경우 sys._MEIPASS를 사용하지 않고, 실행 파일 위치 기준으로 설정
        self.local_working_folder = LOCAL_WORKING_FOLDER

        if not os.path.exists(self.local_working_folder):
            try:
                os.makedirs(self.local_working_folder, exist_ok=True)
            except OSError as e_mkdir:
                logging.critical(f"로컬 작업 폴더({self.local_working_folder}) 생성 실패: {e_mkdir}")
                QMessageBox.critical(None, "치명적 오류", f"작업 폴더 생성 실패: {e_mkdir}\n프로그램을 종료합니다.")
                sys.exit(1)  # 폴더 생성 실패 시 프로그램 종료

        self.google_file_id = "16g4Btk17vNHSTPy-b40kxCESY38g0cD-"  # 고정된 Google Drive 파일 ID
        self.service_account_credential_file = resource_path("service_credentials.json")  # JSON 키 파일 경로
        self.google_api_scopes = ['https://www.googleapis.com/auth/drive']  # Google Drive API 접근 범위

        self.latest_raw_data_file = None  # 가장 최근 다운로드된 원시 데이터 파일 경로
        self.current_operation_save_path = None  # 현재 작업(다운로드/예측)의 저장 경로 (GUI 모드 시 사용자가 선택)

        if not KIHS.objectName():
            KIHS.setObjectName(u"KIHS")
        KIHS.resize(357, 790)
        self.label = QLabel(KIHS)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(0, 0, 357, 790))
        self.label.setStyleSheet(u"background-color:white;")
        self.label_2 = QLabel(KIHS)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(37, 6, 291, 121))
        self.label_2.setScaledContents(True)
        self.pushButton = QPushButton(KIHS)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(50, 280, 101, 41))
        font = QFont()
        font.setFamilies([u"\ud734\uba3c\uc5d1\uc2a4\ud3ec"])
        font.setPointSize(12)
        font.setBold(False)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet(u"background-color:rgba(38, 27, 252, 200);\n"
                                      "color:white;")
        self.label_6 = QLabel(KIHS)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(0, 140, 357, 31))
        font1 = QFont()
        font1.setFamilies([u"\ud55c\ucef4 \ub9d0\ub791\ub9d0\ub791 Bold"])
        font1.setPointSize(16)
        self.label_6.setFont(font1)
        self.label_6.setAlignment(Qt.AlignCenter)
        self.comboBox = QComboBox(KIHS)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        #self.comboBox.addItem("")
        #self.comboBox.addItem("")
        #self.comboBox.addItem("")
        #self.comboBox.addItem("")
        #self.comboBox.addItem("")
        #self.comboBox.addItem("")
        #self.comboBox.addItem("")
        #self.comboBox.addItem("")
        #self.comboBox.addItem("")
        #self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(40, 210, 281, 31))
        font2 = QFont()
        font2.setFamilies([u"\ud55c\ucef4 \ub9d0\ub791\ub9d0\ub791 Bold"])
        font2.setPointSize(10)
        font2.setBold(False)
        self.comboBox.setFont(font2)
        self.comboBox.setStyleSheet(u"background-color: rgb(255, 255, 255);\n"
                                    "selection-color: rgb(0, 0, 0);\n"
                                    "selection-background-color: rgba(85, 170, 255, 200);")
        self.comboBox.setInsertPolicy(QComboBox.InsertAtBottom)
        self.label_8 = QLabel(KIHS)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(40, 179, 281, 31))
        font3 = QFont()
        font3.setFamilies([u"\ud55c\ucef4 \ub9d0\ub791\ub9d0\ub791 Bold"])
        font3.setPointSize(14)
        self.label_8.setFont(font3)
        self.label_8.setStyleSheet(u"background-color: rgb(255, 255, 127);")
        self.label_8.setAlignment(Qt.AlignCenter)
        self.pushButton_2 = QPushButton(KIHS)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(210, 280, 101, 41))
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet(u"background-color:rgba(255, 50, 50, 200);\n"
                                        "color:white;")
        self.pushButton_all = QPushButton(KIHS)
        self.pushButton_all.setObjectName(u"pushButton_all")
        self.pushButton_all.setGeometry(QRect(50, 330, 221, 41))
        self.pushButton_all.setFont(font)
        self.pushButton_all.setStyleSheet(u"background-color:rgba(50, 205, 50, 200);\n"
                                          "color:white;")
        self.pushButton_all.setText(QCoreApplication.translate("KIHS", u"\uc804\uccb4 \uc9c0\uc810 \uc608\uce21", None))
        self.tableWidget = QTableWidget(KIHS)
        if (self.tableWidget.columnCount() < 2):
            self.tableWidget.setColumnCount(2)
        font4 = QFont()
        font4.setFamilies([u"\ud55c\ucef4 \ub9d0\ub791\ub9d0\ub791 Bold"])
        __qtablewidgetitem = QTableWidgetItem()
        __qtablewidgetitem.setFont(font4);
        self.tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        __qtablewidgetitem1.setFont(font4);
        self.tableWidget.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        if (self.tableWidget.rowCount() < 4):
            self.tableWidget.setRowCount(4)
        __qtablewidgetitem2 = QTableWidgetItem()
        __qtablewidgetitem2.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem2.setFont(font4);
        self.tableWidget.setVerticalHeaderItem(0, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        __qtablewidgetitem3.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem3.setFont(font4);
        self.tableWidget.setVerticalHeaderItem(1, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        __qtablewidgetitem4.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem4.setFont(font4);
        self.tableWidget.setVerticalHeaderItem(2, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        __qtablewidgetitem5.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem5.setFont(font4);
        self.tableWidget.setVerticalHeaderItem(3, __qtablewidgetitem5)
        font5 = QFont()
        font5.setFamilies([u"\ud55c\ucef4 \ub9d0\ub791\ub9d0\ub791 Regular"])
        font5.setPointSize(8)
        __qtablewidgetitem6 = QTableWidgetItem()
        __qtablewidgetitem6.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem6.setFont(font5);
        self.tableWidget.setItem(0, 0, __qtablewidgetitem6)
        __qtablewidgetitem7 = QTableWidgetItem()
        __qtablewidgetitem7.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem7.setFont(font5);
        self.tableWidget.setItem(0, 1, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        __qtablewidgetitem8.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem8.setFont(font5);
        self.tableWidget.setItem(1, 0, __qtablewidgetitem8)
        __qtablewidgetitem9 = QTableWidgetItem()
        __qtablewidgetitem9.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem9.setFont(font5);
        self.tableWidget.setItem(1, 1, __qtablewidgetitem9)
        __qtablewidgetitem10 = QTableWidgetItem()
        __qtablewidgetitem10.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem10.setFont(font5);
        self.tableWidget.setItem(2, 0, __qtablewidgetitem10)
        __qtablewidgetitem11 = QTableWidgetItem()
        __qtablewidgetitem11.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem11.setFont(font5);
        self.tableWidget.setItem(2, 1, __qtablewidgetitem11)
        __qtablewidgetitem12 = QTableWidgetItem()
        __qtablewidgetitem12.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem12.setFont(font5);
        self.tableWidget.setItem(3, 0, __qtablewidgetitem12)
        __qtablewidgetitem13 = QTableWidgetItem()
        __qtablewidgetitem13.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem13.setFont(font5);
        self.tableWidget.setItem(3, 1, __qtablewidgetitem13)
        self.tableWidget.setObjectName(u"tableWidget")
        self.tableWidget.setGeometry(QRect(30, 380, 297, 126))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setMaximumSize(QSize(331, 16777215))
        self.tableWidget.setSizeIncrement(QSize(0, 0))
        self.tableWidget.setBaseSize(QSize(0, 0))
        font6 = QFont()
        font6.setFamilies([u"\ud55c\ucef4 \ub9d0\ub791\ub9d0\ub791 Regular"])
        self.tableWidget.setFont(font6)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.setRowCount(4)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(23)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(122)
        self.tableWidget.verticalHeader().setMinimumSectionSize(25)
        self.tableWidget.verticalHeader().setDefaultSectionSize(25)
        self.tableWidget_2 = QTableWidget(KIHS)
        if (self.tableWidget_2.columnCount() < 2):
            self.tableWidget_2.setColumnCount(2)
        __qtablewidgetitem14 = QTableWidgetItem()
        __qtablewidgetitem14.setFont(font4);
        self.tableWidget_2.setHorizontalHeaderItem(0, __qtablewidgetitem14)
        __qtablewidgetitem15 = QTableWidgetItem()
        __qtablewidgetitem15.setFont(font4);
        self.tableWidget_2.setHorizontalHeaderItem(1, __qtablewidgetitem15)
        if (self.tableWidget_2.rowCount() < 7):
            self.tableWidget_2.setRowCount(7)
        __qtablewidgetitem16 = QTableWidgetItem()
        __qtablewidgetitem16.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem16.setFont(font4);
        self.tableWidget_2.setVerticalHeaderItem(0, __qtablewidgetitem16)
        __qtablewidgetitem17 = QTableWidgetItem()
        __qtablewidgetitem17.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem17.setFont(font4);
        self.tableWidget_2.setVerticalHeaderItem(1, __qtablewidgetitem17)
        __qtablewidgetitem18 = QTableWidgetItem()
        __qtablewidgetitem18.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem18.setFont(font4);
        self.tableWidget_2.setVerticalHeaderItem(2, __qtablewidgetitem18)
        __qtablewidgetitem19 = QTableWidgetItem()
        __qtablewidgetitem19.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem19.setFont(font4);
        self.tableWidget_2.setVerticalHeaderItem(3, __qtablewidgetitem19)
        __qtablewidgetitem20 = QTableWidgetItem()
        __qtablewidgetitem20.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem20.setFont(font4);
        self.tableWidget_2.setVerticalHeaderItem(4, __qtablewidgetitem20)
        __qtablewidgetitem21 = QTableWidgetItem()
        __qtablewidgetitem21.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem21.setFont(font4);
        self.tableWidget_2.setVerticalHeaderItem(5, __qtablewidgetitem21)
        __qtablewidgetitem22 = QTableWidgetItem()
        __qtablewidgetitem22.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem22.setFont(font4);
        self.tableWidget_2.setVerticalHeaderItem(6, __qtablewidgetitem22)
        __qtablewidgetitem23 = QTableWidgetItem()
        __qtablewidgetitem23.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem23.setFont(font5);
        self.tableWidget_2.setItem(0, 0, __qtablewidgetitem23)
        __qtablewidgetitem24 = QTableWidgetItem()
        __qtablewidgetitem24.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem24.setFont(font5);
        self.tableWidget_2.setItem(0, 1, __qtablewidgetitem24)
        __qtablewidgetitem25 = QTableWidgetItem()
        __qtablewidgetitem25.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem25.setFont(font5);
        self.tableWidget_2.setItem(1, 0, __qtablewidgetitem25)
        __qtablewidgetitem26 = QTableWidgetItem()
        __qtablewidgetitem26.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem26.setFont(font5);
        self.tableWidget_2.setItem(1, 1, __qtablewidgetitem26)
        __qtablewidgetitem27 = QTableWidgetItem()
        __qtablewidgetitem27.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem27.setFont(font5);
        self.tableWidget_2.setItem(2, 0, __qtablewidgetitem27)
        __qtablewidgetitem28 = QTableWidgetItem()
        __qtablewidgetitem28.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem28.setFont(font5);
        self.tableWidget_2.setItem(2, 1, __qtablewidgetitem28)
        __qtablewidgetitem29 = QTableWidgetItem()
        __qtablewidgetitem29.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem29.setFont(font5);
        self.tableWidget_2.setItem(3, 0, __qtablewidgetitem29)
        __qtablewidgetitem30 = QTableWidgetItem()
        __qtablewidgetitem30.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem30.setFont(font5);
        self.tableWidget_2.setItem(3, 1, __qtablewidgetitem30)
        brush = QBrush(QColor(170, 255, 255, 100))
        brush.setStyle(Qt.SolidPattern)
        __qtablewidgetitem31 = QTableWidgetItem()
        __qtablewidgetitem31.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem31.setFont(font5);
        __qtablewidgetitem31.setBackground(brush)
        self.tableWidget_2.setItem(4, 0, __qtablewidgetitem31)
        brush1 = QBrush(QColor(255, 0, 0, 255))
        brush1.setStyle(Qt.NoBrush)
        brush2 = QBrush(QColor(170, 255, 255, 150))
        brush2.setStyle(Qt.SolidPattern)
        font7 = QFont()
        font7.setFamilies([u"\ud55c\ucef4 \ub9d0\ub791\ub9d0\ub791 Regular"])
        font7.setPointSize(9)
        __qtablewidgetitem32 = QTableWidgetItem()
        __qtablewidgetitem32.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem32.setFont(font7);
        __qtablewidgetitem32.setBackground(brush2)
        __qtablewidgetitem32.setForeground(brush1)
        self.tableWidget_2.setItem(4, 1, __qtablewidgetitem32)
        __qtablewidgetitem33 = QTableWidgetItem()
        __qtablewidgetitem33.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem33.setFont(font5);
        __qtablewidgetitem33.setBackground(brush)
        self.tableWidget_2.setItem(5, 0, __qtablewidgetitem33)
        brush3 = QBrush(QColor(255, 0, 0, 255))
        brush3.setStyle(Qt.NoBrush)
        __qtablewidgetitem34 = QTableWidgetItem()
        __qtablewidgetitem34.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem34.setFont(font7);
        __qtablewidgetitem34.setBackground(brush2)
        __qtablewidgetitem34.setForeground(brush3)
        self.tableWidget_2.setItem(5, 1, __qtablewidgetitem34)
        __qtablewidgetitem35 = QTableWidgetItem()
        __qtablewidgetitem35.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem35.setFont(font5);
        __qtablewidgetitem35.setBackground(brush)
        self.tableWidget_2.setItem(6, 0, __qtablewidgetitem35)
        brush4 = QBrush(QColor(255, 0, 0, 255))
        brush4.setStyle(Qt.NoBrush)
        __qtablewidgetitem36 = QTableWidgetItem()
        __qtablewidgetitem36.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem36.setFont(font7);
        __qtablewidgetitem36.setBackground(brush2)
        __qtablewidgetitem36.setForeground(brush4)
        self.tableWidget_2.setItem(6, 1, __qtablewidgetitem36)
        self.tableWidget_2.setObjectName(u"tableWidget_2")
        self.tableWidget_2.setGeometry(QRect(30, 512, 297, 201))
        sizePolicy.setHeightForWidth(self.tableWidget_2.sizePolicy().hasHeightForWidth())
        self.tableWidget_2.setSizePolicy(sizePolicy)
        self.tableWidget_2.setSizeIncrement(QSize(0, 0))
        self.tableWidget_2.setBaseSize(QSize(0, 0))
        self.tableWidget_2.setFont(font7)
        self.tableWidget_2.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget_2.setRowCount(7)
        self.tableWidget_2.setColumnCount(2)
        self.tableWidget_2.horizontalHeader().setMinimumSectionSize(23)
        self.tableWidget_2.horizontalHeader().setDefaultSectionSize(122)
        self.tableWidget_2.verticalHeader().setMinimumSectionSize(25)
        self.tableWidget_2.verticalHeader().setDefaultSectionSize(25)
        self.textBrowser = QTextBrowser(KIHS)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setGeometry(QRect(20, 717, 316, 61))
        font8 = QFont()
        font8.setFamilies([u"\ud55c\ucef4 \ub9d0\ub791\ub9d0\ub9d0 Regular"])
        font8.setPointSize(8)
        font8.setBold(False)
        font8.setItalic(False)
        self.textBrowser.setFont(font8)
        self.textBrowser.setStyleSheet(u"background-color: rgb(152, 255, 161);\n"
                                       "font: 8pt \"\ud55c\ucef4 \ub9d0\ub791\ub9d0\ub791 Regular\";")
        self.progressBar = QProgressBar(KIHS)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setGeometry(QRect(40, 250, 281, 23))
        self.progressBar.setValue(0)
        self.checkBox_auto = QCheckBox(KIHS)
        self.checkBox_auto.setObjectName("checkBox_auto")
        self.checkBox_auto.setText("주기\n실행\n모드")
        self.checkBox_auto.setGeometry(QRect(285, 325, 150, 50))
        self.retranslateUi(KIHS)
        self.pushButton.clicked.connect(self.download_clicked)
        self.pushButton_2.clicked.connect(self.on_pushButton2_clicked)
        self.pushButton_all.clicked.connect(self.on_pushButton_all_clicked)

        # 이미지 경로 설정
        if getattr(sys, 'frozen', False):
            # 실행 파일로 실행되는 경우
            bundle_dir = sys._MEIPASS
        else:
            # 스크립트로 실행되는 경우
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(bundle_dir, 'CI.jpg')
        icon_path = os.path.join(bundle_dir, 'ICON.ico')
        KIHS.setWindowIcon(QIcon(icon_path))
        self.label_2.setPixmap(QPixmap(image_path))

        self._auto_timer = QTimer(KIHS)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.timeout.connect(self.download_clicked)
        self.checkBox_auto.stateChanged.connect(self._on_auto_toggled)
        self._download_has_errors = False

        QMetaObject.connectSlotsByName(KIHS)

    def retranslateUi(self, KIHS):
        KIHS.setWindowTitle(QCoreApplication.translate("KIHS",
                                                       u"\u0041\u002e\u0049\u002e\u0020\ub525\ub7ec\ub2dd\u0020\uae30\ubc18\u0020\ud64d\uc218\uc704\u0020\uc608\uce21\u0020\u0056\u0032\u0035\u0030\u0035\u0031\u0039",
                                                       None))
        self.label.setText("")
        self.label_2.setText("")
        self.pushButton.setText(QCoreApplication.translate("KIHS", u"\uc790\ub8cc\uc218\uc9d1", None))
        self.label_6.setText(QCoreApplication.translate("KIHS",
                                                        u"\u0041\u002e\u0049\u002e\u0020\ub525\ub7ec\ub2dd\u0020\uae30\ubc18\u0020\ud64d\uc218\uc704\u0020\uc608\uce21",
                                                        None))
        self.comboBox.setItemText(0,
                                  QCoreApplication.translate("KIHS", u"1002698_\uc601\uc6d4\uad70(\ud314\uad34\uad50)",
                                                             None))
        self.comboBox.setItemText(1,
                                  QCoreApplication.translate("KIHS", u"1006690_\uc6d0\uc8fc\uc2dc(\ubb38\ub9c9\uad50)",
                                                             None))
        self.comboBox.setItemText(2, QCoreApplication.translate("KIHS",
                                                                u"1007625_\uc5ec\uc8fc\uc2dc(\ub0a8\ud55c\uac15\uad50)",
                                                                None))
        self.comboBox.setItemText(3,
                                  QCoreApplication.translate("KIHS", u"1014650_\ud64d\ucc9c\uad70(\ud64d\ucc9c\uad50)",
                                                             None))
        self.comboBox.setItemText(4,
                                  QCoreApplication.translate("KIHS", u"1014680_\ud64d\ucc9c\uad70(\ubc18\uace1\uad50)",
                                                             None))
        self.comboBox.setItemText(5,
                                  QCoreApplication.translate("KIHS", u"1016670_\uad11\uc8fc\uc2dc(\uc11c\ud558\uad50)",
                                                             None))
        self.comboBox.setItemText(6, QCoreApplication.translate("KIHS",
                                                                u"1101630_\ucc9c\uc548\uc2dc(\uc548\uc131\ucc9c\uad50)",
                                                                None))
        self.comboBox.setItemText(7,
                                  QCoreApplication.translate("KIHS", u"3009693_\ub300\uc804\uc2dc(\uc2e0\uad6c\uad50)",
                                                             None))
        self.comboBox.setItemText(8,
                                  QCoreApplication.translate("KIHS", u"3010660_\uc138\uc885\uc2dc(\uba85\ud559\ub9ac)",
                                                             None))
        self.comboBox.setItemText(9, QCoreApplication.translate("KIHS",
                                                                u"3011665_\uccad\uc8fc\uc2dc(\ubbf8\ud638\uac15\uad50)",
                                                                None))
        self.comboBox.setItemText(10,
                                  QCoreApplication.translate("KIHS", u"3011685_\uc138\uc885\uc2dc(\ubbf8\ud638\uad50)",
                                                             None))
        self.comboBox.setItemText(11,
                                  QCoreApplication.translate("KIHS", u"3302658_\uc815\uc74d\uc2dc(\ucd08\uac15\ub9ac)",
                                                             None))
        self.comboBox.setItemText(12,
                                  QCoreApplication.translate("KIHS", u"4004615_\uc21c\ucc3d\uad70(\uc720\uc801\uad50)",
                                                             None))
        self.comboBox.setItemText(13,
                                  QCoreApplication.translate("KIHS", u"4004660_\ub0a8\uc6d0\uc2dc(\uc2e0\ub355\ub9ac)",
                                                             None))
        self.comboBox.setItemText(14,
                                  QCoreApplication.translate("KIHS", u"4004690_\uace1\uc131\uad70(\uae08\uace1\uad50)",
                                                             None))
        self.comboBox.setItemText(15,
                                  QCoreApplication.translate("KIHS", u"4006660_\uace1\uc131\uad70(\uace0\ub2ec\uad50)",
                                                             None))
        self.comboBox.setItemText(16,
                                  QCoreApplication.translate("KIHS", u"4008670_\uace1\uc131\uad70(\ud0dc\uc548\uad50)",
                                                             None))
        self.comboBox.setItemText(17,
                                  QCoreApplication.translate("KIHS", u"4009610_\uad6c\ub840\uad70(\uad6c\ub840\uad50)",
                                                             None))
        self.comboBox.setItemText(18,
                                  QCoreApplication.translate("KIHS", u"4009630_\uad6c\ub840\uad70(\uc1a1\uc815\ub9ac)",
                                                             None))
        self.comboBox.setItemText(19, QCoreApplication.translate("KIHS",
                                                                 u"4009640_\uad11\uc591\uc2dc(\ub0a8\ub3c4\ub300\uad50)",
                                                                 None))
        self.comboBox.setItemText(20, QCoreApplication.translate("KIHS",
                                                                 u"5001640_\uad11\uc8fc\uad11\uc5ed\uc2dc(\uc6a9\uc0b0\uad50)",
                                                                 None))
        self.comboBox.setItemText(21, QCoreApplication.translate("KIHS",
                                                                 u"5001645_\uad11\uc8fc\uad11\uc5ed\uc2dc(\ucca8\ub2e8\ub300\uad50)",
                                                                 None))
        # self.comboBox.setItemText(22, QCoreApplication.translate("KIHS", u"1018630_\ub0a8\uc591\uc8fc\uc2dc(\uc9c4\uad00\uad50)", None))
        # self.comboBox.setItemText(23, QCoreApplication.translate("KIHS", u"1018638_\ub0a8\uc591\uc8fc\uc2dc(\uc655\uc219\uad50)", None))
        # self.comboBox.setItemText(24, QCoreApplication.translate("KIHS", u"1018675_\uc11c\uc6b8\uc2dc(\uc911\ub791\uad50)", None))
        # self.comboBox.setItemText(25, QCoreApplication.translate("KIHS", u"3008670_\uae08\uc0b0\uad70(\uc81c\uc6d0\uad50)", None))
        # self.comboBox.setItemText(26, QCoreApplication.translate("KIHS", u"3009680_\ub300\uc804\uc2dc(\uc6d0\ucd0c\uad50)", None))
        # self.comboBox.setItemText(27, QCoreApplication.translate("KIHS", u"3011645_\uccad\uc8fc\uc2dc(\ud765\ub355\uad50)", None))
        # self.comboBox.setItemText(28, QCoreApplication.translate("KIHS", u"3301654_\uc804\uc8fc\uc2dc(\uc138\ub0b4\uad50)", None))
        # self.comboBox.setItemText(29, QCoreApplication.translate("KIHS", u"3301665_\uc804\uc8fc\uc2dc(\ubbf8\uc0b0\uad50)", None))
        # self.comboBox.setItemText(30, QCoreApplication.translate("KIHS", u"3302645_\uc815\uc74d\uc2dc(\uc8fd\ub9bc\uad50)", None))
        # self.comboBox.setItemText(31, QCoreApplication.translate("KIHS", u"5001660_\uad11\uc8fc\uad11\uc5ed\uc2dc(\uc5b4\ub4f1\ub300\uad50)", None))
        self.label_8.setText(
            QCoreApplication.translate("KIHS", u"\uc608\uce21 \uc218\uc704\uad00\uce21\uc18c \uc120\ud0dd", None))
        self.pushButton_2.setText(QCoreApplication.translate("KIHS", u"\uc218\uc704\uc608\uce21", None))
        self.pushButton_all.setText(QCoreApplication.translate("KIHS", u"전체 지점 예측", None))
        ___qtablewidgetitem = self.tableWidget.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("KIHS", u"\uc2e4\uc81c \uc2dc\uac04", None))
        ___qtablewidgetitem1 = self.tableWidget.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("KIHS", u"\uc2e4\uc81c \uc218\uc704", None))
        ___qtablewidgetitem2 = self.tableWidget.verticalHeaderItem(0)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("KIHS", u"3\uc2dc\uac04 \uc804", None))
        ___qtablewidgetitem3 = self.tableWidget.verticalHeaderItem(1)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("KIHS", u"2\uc2dc\uac04 \uc804", None))
        ___qtablewidgetitem4 = self.tableWidget.verticalHeaderItem(2)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("KIHS", u"1\uc2dc\uac04 \uc804", None))
        ___qtablewidgetitem5 = self.tableWidget.verticalHeaderItem(3)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("KIHS", u"\ud604\uc7ac\uc2dc\uac04", None))

        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.setSortingEnabled(__sortingEnabled)

        ___qtablewidgetitem6 = self.tableWidget_2.horizontalHeaderItem(0)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("KIHS", u"\uc608\uce21 \uc2dc\uac04", None))
        ___qtablewidgetitem7 = self.tableWidget_2.horizontalHeaderItem(1)
        ___qtablewidgetitem7.setText(QCoreApplication.translate("KIHS", u"\uc608\uce21 \uc218\uc704", None))
        ___qtablewidgetitem8 = self.tableWidget_2.verticalHeaderItem(0)
        ___qtablewidgetitem8.setText(QCoreApplication.translate("KIHS", u"3\uc2dc\uac04 \uc804", None))
        ___qtablewidgetitem9 = self.tableWidget_2.verticalHeaderItem(1)
        ___qtablewidgetitem9.setText(QCoreApplication.translate("KIHS", u"2\uc2dc\uac04 \uc804", None))
        ___qtablewidgetitem10 = self.tableWidget_2.verticalHeaderItem(2)
        ___qtablewidgetitem10.setText(QCoreApplication.translate("KIHS", u"1\uc2dc\uac04 \uc804", None))
        ___qtablewidgetitem11 = self.tableWidget_2.verticalHeaderItem(3)
        ___qtablewidgetitem11.setText(QCoreApplication.translate("KIHS", u"\ud604\uc7ac\uc2dc\uac04", None))
        ___qtablewidgetitem12 = self.tableWidget_2.verticalHeaderItem(4)
        ___qtablewidgetitem12.setText(QCoreApplication.translate("KIHS", u"1\uc2dc\uac04 \ud6c4", None))
        ___qtablewidgetitem13 = self.tableWidget_2.verticalHeaderItem(5)
        ___qtablewidgetitem13.setText(QCoreApplication.translate("KIHS", u"2\uc2dc\uac04 \ud6c4", None))
        ___qtablewidgetitem14 = self.tableWidget_2.verticalHeaderItem(6)
        ___qtablewidgetitem14.setText(QCoreApplication.translate("KIHS", u"3\uc2dc\uac04 \ud6c4", None))

        __sortingEnabled1 = self.tableWidget_2.isSortingEnabled()
        self.tableWidget_2.setSortingEnabled(False)
        self.tableWidget_2.setSortingEnabled(__sortingEnabled1)

        self.textBrowser.setHtml(QCoreApplication.translate("KIHS",
                                                            u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n" "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n" "p, li { white-space: pre-wrap; }\n" "hr { height: 1px; border-width: 0; }\n" "li.unchecked::marker { content: \"\\2610\"; }\n" "li.checked::marker { content: \"\\2612\"; }\n" "</style></head><body style=\" font-family:'\ud55c\ucef4 \ub9d0\ub791\ub9d0\ub791 Regular'; font-size:8pt; font-weight:400; font-style:normal;\">\n" "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">\u203b \ud574\ub2f9 \uc790\ub8cc\ub294 \uc778\uacf5\uc9c0\ub2a5</span><span style=\" font-size:10pt; color:#0000ff;\"> \ub525\ub7ec\ub2dd</span><span style=\" font-size:10pt;\">\uc744 \ud1b5\ud574 </span><span style=\" font-size:10pt; color:#ff0000;\">\ud559\uc2b5\ub41c \ub370\uc774\ud130</span><span style=\" font-s" "ize:10pt;\">\ub97c</span></p>\n" "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">\uae30\ubc18\uc73c\ub85c</span><span style=\" font-size:10pt; color:#0000ff;\"> 3\uc2dc\uac04 \uc774\ud6c4</span><span style=\" font-size:10pt;\">\uc758 \uc218\uc704\ub97c </span><span style=\" font-size:10pt; color:#ff0000;\">\uc608\uce21</span><span style=\" font-size:10pt;\">\ud55c \ubaa8\ub378\ub9c1 \uc790\ub8cc\ub85c, </span></p>\n" "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">\uc2e4\uc81c \uc218\uc704\uc640 </span><span style=\" font-size:10pt; color:#ff0000;\">\ucc28\uc774\uac00 \ubc1c\uc0dd</span><span style=\" font-size:10pt;\">\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4.</span></p></body></html>",
                                                            None))

    # retranslateUi

    def on_worker_error(self, error_message_str: str):
        """DownloadThread에서 오류 발생 시 호출되는 슬롯."""
        logging.error(f"다운로드 스레드 오류: {error_message_str}")
        if not self.checkBox_auto.isChecked():  # GUI 모드에서만 메시지 박스 표시
            QMessageBox.critical(None, "다운로드 오류", error_message_str)
        # 자동 실행 모드에서는 플래그 설정
        if self.checkBox_auto.isChecked():
            self._download_has_errors = True

    def upload_to_google_drive_direct_api(self, local_filepath_to_upload, gdrive_file_id_to_update):
        """
        지정된 로컬 파일을 Google Drive의 특정 파일 ID로 업로드(업데이트).
        서비스 계정 인증 사용.
        """
        try:
            if not os.path.exists(self.service_account_credential_file):
                error_log_msg = f"서비스 계정 인증 파일({self.service_account_credential_file})을 찾을 수 없습니다."
                logging.error(error_log_msg)
                if not self.checkBox_auto.isChecked():
                    QMessageBox.critical(None, "Google Drive 인증 오류", error_log_msg)
                return False  # 업로드 실패

            # 서비스 계정 자격 증명 로드
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_credential_file, scopes=self.google_api_scopes)
            # Google Drive 서비스 빌드
            drive_service = build('drive', 'v3', credentials=credentials)

            # 업로드할 파일 미디어 객체 생성
            file_media_body = MediaFileUpload(local_filepath_to_upload,
                                              mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                              # Excel 파일 타입
                                              resumable=True)

            # 기존 파일 업데이트 요청
            updated_file_metadata = drive_service.files().update(
                fileId=gdrive_file_id_to_update,
                media_body=file_media_body,
                fields='id, name, modifiedTime'  # 응답으로 받을 메타데이터 필드
            ).execute()

            success_log_msg = (f"파일 '{updated_file_metadata.get('name')}'이(가) "
                               f"Google Drive에 성공적으로 업데이트되었습니다. (수정 시간: {updated_file_metadata.get('modifiedTime')})")
            logging.info(success_log_msg)
            if not self.checkBox_auto.isChecked():  # GUI 모드에서만 성공 알림
                QMessageBox.information(None, "Google Drive 업로드 성공",
                                        f"'{os.path.basename(local_filepath_to_upload)}' 파일이 Google Drive에 성공적으로 업데이트되었습니다.")
            return True  # 업로드 성공

        except Exception as e_gdrive_upload:
            error_log_msg = f"Google Drive 업로드/업데이트 중 오류 발생: {e_gdrive_upload}"
            logging.error(error_log_msg, exc_info=True)
            if not self.checkBox_auto.isChecked():  # GUI 모드에서만 오류 알림
                QMessageBox.critical(None, "Google Drive 업로드 오류",
                                     f"Google Drive 업로드 중 오류가 발생했습니다:\n{str(e_gdrive_upload)}")
            return False  # 업로드 실패

    def _on_auto_toggled(self, state_int):
        if Qt.CheckState(state_int) == Qt.Checked:
            self.download_clicked()
            self._auto_timer.setInterval(30 * 60 * 1000)   # 이후 30분(1800_000ms)마다 반복
            self._auto_timer.start()
        else:
            logging.info("자동 주기 실행 모드 비활성화됨.")
            if hasattr(self, '_auto_timer') and self._auto_timer.isActive():
                self._auto_timer.stop()

    def _schedule_next_cycle(self):
        if not self.checkBox_auto.isChecked():
            return
        now = datetime.now()
        candidate = now.replace(minute=10, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(hours=1)
        next_run_time = candidate

        wait_ms = int((next_run_time - now).total_seconds() * 1000)
        if wait_ms < 0:
            wait_ms = 1000 * 60 * 60
            logging.warning(f"다음 실행 시간 계산 오류. 1시간 후로 강제 설정: {next_run_time}")

        logging.info(f"다음 자동 실행 예약: {next_run_time} (대기 시간: {wait_ms / 1000 / 60:.2f}분)")

        if hasattr(self, '_auto_timer'):
            self._auto_timer.start(wait_ms)
        else:
            logging.error("자동 실행 타이머 객체(self._auto_timer)가 초기화되지 않았습니다.")

    def download_clicked(self):
        now = datetime.now()
        end_time_dt = now.replace(minute=0, second=0, microsecond=0)
        start_time_dt = end_time_dt - timedelta(days=10)

        current_save_path = ""
        if self.checkBox_auto.isChecked():
            current_save_path = self.local_working_folder  # 클래스 변수 사용
        else:
            selected_dir = QFileDialog.getExistingDirectory(None, "다운로드 자료를 저장할 폴더 선택", self.local_working_folder)
            if not selected_dir:
                QMessageBox.information(None, "알림", "저장 폴더 선택이 취소되었습니다.")
                return
            current_save_path = selected_dir

        self.current_operation_save_path = current_save_path  # 다음 단계에서 사용할 경로 저장

        # DownloadThread 인스턴스 생성 및 시작
        self.download_thread_instance = DownloadThread(
            self.api_key,  # api_key 전달
            self.observation_codes,
            self.observation2_codes,
            start_time_dt,
            end_time_dt,
            current_save_path,  # 이 인자는 스레드 내에서 현재 직접 사용되지 않음
            max_workers=2
        )
        self.download_thread_instance.errorOccurred.connect(self.on_worker_error)
        self.download_thread_instance.progress_updated.connect(self.update_progress)
        self.download_thread_instance.download_completed.connect(self.on_download_completed)
        self.download_thread_instance.start()

        self.progressBar.setValue(0)
        if not self.checkBox_auto.isChecked():
            QMessageBox.information(None, "다운로드 시작", "기초자료 다운로드를 시작합니다.\n완료 시 알림이 표시됩니다.")
            logging.info(f"자료수집 시작됨. 저장 경로: {current_save_path}")

    def update_progress(self, progress_value: int):
        self.progressBar.setValue(progress_value)

    def on_download_completed(self, all_data):
        # --- 자동 실행 모드, 다운로드 중 에러 발생했으면 10초 뒤 재시도 ---
        is_automated = self.checkBox_auto.isChecked() or getattr(self, 'is_headless', False)
        if is_automated and getattr(self, '_download_has_errors', False):
            logging.warning("기초자료 다운로드 중 일부 오류 발생, 10초 후 재시도합니다.")
            # 재시도 플래그 초기화
            self._download_has_errors = False
            # 10초(10,000ms) 후에 download_clicked() 호출
            QTimer.singleShot(10_000, self.download_clicked)
            return
        if not self.current_operation_save_path:
            logging.error("on_download_completed: current_operation_save_path가 설정되지 않았습니다.")
            return

        end_time_str = datetime.now().strftime("%Y-%m-%d %H")

        raw_data_filename_str = f"Rawdata.xlsx"
        full_raw_filepath = os.path.join(self.current_operation_save_path, raw_data_filename_str)

        observation_codes = ['1002685', '1002687', '1002698', '1005697', '1006670', '1006672', '1006680', '1006690',
                             '1007615', '1007620', '1007625', '1014630', '1014640', '1014650', '1014665', '1014680',
                             '1016607', '1016650', '1016660', '1016670', '1016695', '1018620', '1018625', '1018630',
                             '1018635', '1018638', '1018665', '1018669', '1018670', '1018675', '1101610', '1101620',
                             '1101630', '1101635', '3004637', '3004645', '3008670', '3008695', '3009665', '3009670',
                             '3009673', '3009675', '3009680', '3009693', '3009698', '3010620', '3010660', '3011625',
                             '3011630', '3011635', '3011641', '3011643', '3011645', '3011660', '3011665', '3011685',
                             '3011695', '3012602', '3301652', '3301654', '3301655', '3301657', '3301665', '3302643',
                             '3302645', '3302653', '3302658', '4002640', '4002690', '4003690', '4004615', '4004660',
                             '4004690', '4005670', '4005690', '4006660', '4008660', '4008670', '4009610', '4009630',
                             '4009640', '5001615', '5001620', '5001625', '5001627', '5001640', '5001645', '5001655',
                             '5001660']
        observation2_codes = ['10024060', '10024170', '10024200', '10024220', '10024260', '10054010', '10054020',
                              '10064020', '10064030', '10064050', '10064060', '10064070', '10064080', '10064120',
                              '10074030', '10074100', '10144050', '10144060', '10144070', '10144080', '10144165',
                              '10164010', '10164030', '10164050', '10164075', '10164080', '10184010', '10184110',
                              '10184120', '10184140', '10184190', '10184220', '10224070', '11014020', '11014050',
                              '11014080', '11014110', '11014120', '11014130', '30014010', '30014040', '30014080',
                              '30014140', '30014160', '30034010', '30034020', '30044030', '30084050', '30084070',
                              '30084080', '30094020', '30094040', '30094050', '30104010', '30114020', '30114030',
                              '30114040', '30114060', '30114070', '30114100', '33014070', '33014100', '33024070',
                              '33024080', '40014050', '40014060', '40014070', '40024020', '40024030', '40024050',
                              '40024060', '40034010', '40034020', '40034030', '40044010', '40044030', '40044040',
                              '40044060', '40044070', '40054020', '40054030', '40054040', '40064010', '40074050',
                              '40074060', '40074070', '40074080', '40074082', '40074140', '40074143', '40084010',
                              '40084020', '40094060', '40094070', '40094080', '40094110', '40094120', '40094150',
                              '40094160', '40094170', '50014020', '50014030', '50014050', '50014060', '50014070',
                              '50024020']
        final_dfs = []
        dam_cd_list = ['4001110', '4007110']
        start_date = datetime.now() - timedelta(days=10)
        end_date = datetime.now()
        for damcd in dam_cd_list:
            final_df = download_and_process_dam_data(damcd, start_date, end_date, is_automated=is_automated)
            if not final_df.empty:
                final_dfs.append(final_df)

        combined_final_df = pd.concat(final_dfs, axis=1)
        self.save_to_excel(all_data, full_raw_filepath, observation_codes, observation2_codes, combined_final_df)

        self.latest_raw_data_file = full_raw_filepath

        if is_automated:
            logging.info(f"자동 실행: 기초자료 다운로드 완료 - {full_raw_filepath}")
            # GUI 모드면 comboBox, 헤드리스면 모델 변수 전체 키
            if hasattr(self, 'comboBox'):
                all_codes = list(self.STATION_NAME_MAP.keys())
            else:
                all_codes = list(self.model_variables.keys())
            self.predict_all_locations(all_codes, self.latest_raw_data_file, self.local_working_folder)
        else:
            QMessageBox.information(None, "다운로드 완료", f"통합 자료가 다음 경로에 저장되었습니다:\n'{full_raw_filepath}'")

    def save_to_excel(self, all_data, filename, observation_codes, observation2_codes, final_df):

        all_times = sorted({time for data in all_data.values() for time, _ in data})
        base_df = pd.DataFrame(all_times, columns=['일시'])
        base_df['일시'] = pd.to_datetime(base_df['일시'])

        columns_order = ['일시'] + observation2_codes + observation_codes
        for code in all_data.keys():
            df = pd.DataFrame(all_data[code], columns=['일시', code])
            df['일시'] = pd.to_datetime(df['일시'])
            clean_column_name = code.split('_')[-1]
            df.rename(columns={code: clean_column_name}, inplace=True)
            base_df = pd.merge(base_df, df, on='일시', how='left')

        base_df.set_index('일시', inplace=True)
        base_df.infer_objects(copy=False)
        base_df.reset_index(inplace=True)
        final_columns_order = ['일시'] + [code for code in columns_order[1:]]
        base_df = base_df[final_columns_order]
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        merged_df = pd.merge(base_df, final_df, on='일시', how='left')
        for col in merged_df.columns:
            if col == '일시':
                continue
            try:
                merged_df[col] = pd.to_numeric(merged_df[col])
            except (ValueError, TypeError):
                pass
        num_cols = merged_df.select_dtypes(include='number').columns
        merged_df[num_cols] = (merged_df[num_cols].interpolate(method='linear', limit_direction='forward', axis=0))
        merged_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
        merged_df.to_excel(filename, index=False)

    def on_pushButton2_clicked(self):
        if self.checkBox_auto.isChecked():  # 자동 실행 모드에서는 이 기능 비활성화
            QMessageBox.information(None, "모드 알림", "주기 실행 모드에서는 개별 예측을 지원하지 않습니다.\n"
                                                   "'전체 지점 예측'을 사용하거나, 모드를 해제 후 이용해주세요.")
            return

        target = self.comboBox.currentText()[:7]
        model_files_3h = {
            '1002698': '1002698_model_2503_3h_T1.h5',
            '1006690': '1006690_model_2503_3h_T1.h5',
            '1007625': '1007625_model_2503_3h_T1.h5',
            '1014650': '1014650_model_2503_3h_T1.h5',
            '1014680': '1014680_model_2503_3h_T2.h5',  # T2:past24
            '1016670': '1016670_model_2503_3h_T1.h5',
            # '1018630': '1018630_model_2503_3h_T1.h5',
            # '1018638': '1018638_model_2503_3h_T1.h5',
            # '1018675': '1018675_model_2503_3h_T1.h5',
            '1101630': '1101630_model_2503_3h_T2.h5',  # T2:past24
            # '3008670': '3008670_model_2503_3h_T2.h5', #T2:past24
            # '3009680': '3009680_model_2503_3h_T2.h5', #T2:past24
            '3009693': '3009693_model_2503_3h_T1.h5',
            '3010660': '3010660_model_2503_3h_T1.h5',
            # '3011645': '3011645_model_2503_3h_T1.h5',
            '3011665': '3011665_model_2503_3h_T1.h5',
            '3011685': '3011685_model_2503_3h_T1.h5',
            # '3301654': '3301654_model_241111_3h_T3.h5',
            # '3301665': '3301665_model_241111_3h_T3.h5',
            # '3302645': '3302645_model_241111_3h_T2.h5',
            '3302658': '3302658_model_241111_3h_T2.h5',
            '4004615': '4004615_model_241111_3h_T2.h5',
            '4004660': '4004660_model_241111_3h_T2.h5',
            '4004690': '4004690_model_241111_3h_T2.h5',
            '4006660': '4006660_model_241111_3h_T2.h5',
            '4008670': '4008670_model_241111_3h_T2.h5',
            '4009610': '4009610_model_241111_3h_T2.h5',
            '4009630': '4009630_model_241111_3h_T2.h5',
            '4009640': '4009640_model_241111_3h_T2.h5',
            '5001640': '5001640_model_241111_3h_T2.h5',
            '5001645': '5001645_model_241111_3h_T2.h5',
            # '5001660': '5001660_model_241111_3h_T2.h5',
        }

        model_files_6h = {
            '1002698': '1002698_model_2503_6h_T1.h5',
            '1006690': '1006690_model_2503_6h_T1.h5',
            '1007625': '1007625_model_2503_6h_T1.h5',
            '1014650': '1014650_model_2503_6h_T1.h5',
            '1014680': '1014680_model_2503_6h_T2.h5',  # T2:past24
            '1016670': '1016670_model_2503_6h_T1.h5',
            # '1018630': '1018630_model_2503_6h_T1.h5',
            # '1018638': '1018638_model_2503_6h_T1.h5',
            # '1018675': '1018675_model_2503_6h_T1.h5',
            '1101630': '1101630_model_2503_6h_T2.h5',  # T2:past24
            # '3008670': '3008670_model_2503_6h_T2.h5', #T2:past24
            # '3009680': '3009680_model_2503_6h_T2.h5', #T2:past24
            '3009693': '3009693_model_2503_6h_T2.h5',  # T2:past24
            '3010660': '3010660_model_2503_6h_T1.h5',
            # '3011645': '3011645_model_2503_6h_T1.h5',
            '3011665': '3011665_model_2503_6h_T1.h5',
            '3011685': '3011685_model_2503_6h_T1.h5',
            # '3301654': '3301654_model_241111_6h_T3.h5',
            # '3301665': '3301665_model_241111_6h_T3.h5',
            # '3302645': '3302645_model_241111_6h_T2.h5',
            '3302658': '3302658_model_241111_6h_T2.h5',
            '4004615': '4004615_model_241111_6h_T2.h5',
            '4004660': '4004660_model_241111_6h_T2.h5',
            '4004690': '4004690_model_241111_6h_T2.h5',
            '4006660': '4006660_model_241111_6h_T2.h5',
            '4008670': '4008670_model_241111_6h_T2.h5',
            '4009610': '4009610_model_241111_6h_T2.h5',
            '4009630': '4009630_model_241111_6h_T2.h5',
            '4009640': '4009640_model_241111_6h_T2.h5',
            '5001640': '5001640_model_241111_6h_T2.h5',
            '5001645': '5001645_model_241111_6h_T2.h5',
            # '5001660': '5001660_model_241111_6h_T2.h5',
        }

        if target not in model_files_3h or target not in model_files_6h:
            QMessageBox.information(None, "오류", f"선택한 관측소 '{target}'에 대한 모델 파일이 없습니다.")
            return

        # 모델 파일 경로를 가져옴
        model_file_3h = model_files_3h[target]
        model_file_6h = model_files_6h[target]

        # 모델 파일 경로 수정
        if getattr(sys, 'frozen', False):
            bundle_dir = sys._MEIPASS
        else:
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        model_path_3h = resource_path(os.path.join('models', model_file_3h))
        model_path_6h = resource_path(os.path.join('models', model_file_6h))
        self.process_and_predict2(model_path_3h, model_path_6h, self.model_variables)

    def process_and_predict2(self, model_path_3h, model_path_6h, model_variables):
        target = self.comboBox.currentText()[:7]
        try:
            if target not in model_variables:
                QMessageBox.information(None, "오류", f"선택한 관측소 '{target}'에 대한 정보가 없습니다.")
                return
            selected_variables = model_variables[target]

            data_file, _ = QFileDialog.getOpenFileName(None, "기초 데이터를 업로드해주세요.", "", "Excel files (*.xlsx *.xls)")
            if not data_file:
                QMessageBox.information(None, "알림", "데이터 파일이 선택되지 않았습니다.")
                return

            predictor = Predictor()

            folder_path = QFileDialog.getExistingDirectory(None, "결과를 저장할 폴더 선택")
            if not folder_path:
                QMessageBox.information(None, "알림", "저장 폴더 선택이 취소되었습니다.")
                return

            timestamp = datetime.now().strftime("%m-%d %H")
            # 데이터 로드
            data = pd.read_excel(data_file)
            data['일시'] = pd.to_datetime(data['일시'])
            actual_data = data[[target]]

            # 3시간 뒤 예측
            past3 = self.get_past_hours(target, future_hours=3)
            comparison_3h = predictor.process_and_predict(
                model_path_3h, model_variables, data_file,
                target, past3, future=3
            )

            # 6시간 뒤 예측
            past6 = self.get_past_hours(target, future_hours=6)
            comparison_6h = predictor.process_and_predict(
                model_path_6h, model_variables, data_file,
                target, past6, future=6
            )

            # 실제 데이터와 예측 결과 병합
            merged_data = data[['일시', target]].copy()

            # 예측 결과 병합
            merged_data = pd.merge(merged_data, comparison_3h, on='일시', how='outer')
            merged_data = pd.merge(merged_data, comparison_6h, on='일시', how='outer')
            merged_data.sort_values(by='일시', inplace=True)
            merged_data.reset_index(drop=True, inplace=True)

            # 병합된 데이터 저장
            target_label = self.comboBox.currentText()
            result_file = f"{target_label}_prediction_{timestamp}.xlsx"
            full_file_path = os.path.join(folder_path, result_file)
            merged_data.to_excel(full_file_path, index=False)

            QMessageBox.information(None, "알림", f"예측 결과 파일이 저장되었습니다.")
            target_label = self.comboBox.currentText()
            self.display_predictions(merged_data, target, folder_path)

        except Exception as e:
            QMessageBox.critical(None, "Error", f"예측 중 오류가 발생했습니다: {str(e)}")

    def on_pushButton_all_clicked(self):
        all_targets = [self.comboBox.itemText(i)[:7] for i in range(self.comboBox.count())]

        input_raw_data_filepath = ""
        output_results_folder = ""

        if self.checkBox_auto.isChecked():  # 자동 실행 모드
            if not self.latest_raw_data_file or not os.path.exists(self.latest_raw_data_file):
                logging.error("자동 실행 모드(전체 예측): 기초자료 파일 경로가 유효하지 않습니다. 자료수집을 먼저 실행해야 합니다.")
                QMessageBox.warning(None, "자동 실행 오류", "자동 전체 예측을 위한 기초자료 파일이 없습니다.\n자료수집이 먼저 실행되어야 합니다.")
                # 자동 모드에서는 여기서 다음 주기를 스케줄링하지 않고 종료 (오류 상황)
                return
            input_raw_data_filepath = self.latest_raw_data_file
            output_results_folder = self.local_working_folder
        else:  # GUI 모드
            default_browse_dir_all = os.path.dirname(
                self.latest_raw_data_file) if self.latest_raw_data_file else self.local_working_folder
            input_raw_data_filepath, _ = QFileDialog.getOpenFileName(
                None, "전체 예측에 사용할 통합 데이터 파일(Rawdata.xlsx) 선택",
                default_browse_dir_all, "Excel files (*.xlsx *.xls)")
            if not input_raw_data_filepath:
                QMessageBox.information(None, "알림", "기초 데이터 파일이 선택되지 않았습니다.")
                return
            output_results_folder = QFileDialog.getExistingDirectory(
                None, "전체 예측 결과를 저장할 폴더 선택", self.local_working_folder)
            if not output_results_folder:
                QMessageBox.information(None, "알림", "결과 저장 폴더 선택이 취소되었습니다.")
                return

        self.predict_all_locations(all_targets, input_raw_data_filepath, output_results_folder)

    def predict_all_locations(self, targets, raw_data_path, output_folder):
        is_auto = self.checkBox_auto.isChecked()
        try:
            data = pd.read_excel(raw_data_path)
            data['일시'] = pd.to_datetime(data['일시'])
        except Exception as e:
            msg = f"전체 예측: 기초 데이터 파일 로드 오류 ({raw_data_path}): {e}" # 경로 정보 추가
            logging.error(msg, exc_info=True)
            if not is_auto:
                QMessageBox.critical(None, "파일 오류", msg)
            # 자동 모드 오류 시 다음 주기 스케줄링 중단 또는 별도 처리 필요
            if is_auto and hasattr(self, '_auto_timer') and self._auto_timer.isActive():
                logging.info("기초 데이터 로드 오류로 자동 주기 실행 중단.")
                self._auto_timer.stop()
            return

        if is_auto:
            out_filename = "All_Locations_Prediction.xlsx"
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_filename = f"All_Locations_Prediction_{timestamp}.xlsx"
        out_path = os.path.join(output_folder, out_filename)

        predictor = Predictor()
        try: # ExcelWriter 사용 시 예외 처리 추가
            with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
                for target in targets:
                    if target not in self.model_variables:
                        continue
                    selected_variables = self.model_variables[target]
                    model_path_3h = self.get_model_files(target, future_hours=3)
                    model_path_6h = self.get_model_files(target, future_hours=6)

                    if not model_path_3h or not model_path_6h:
                        continue

                    # 3시간 뒤 예측
                    past3 = self.get_past_hours(target, future_hours=3)
                    comparison_3h = predictor.process_and_predict(
                        model_path_3h, self.model_variables, raw_data_path,
                        target, past3, future=3, is_automated_run=is_auto
                    )

                    # 6시간 뒤 예측
                    past6 = self.get_past_hours(target, future_hours=6)
                    comparison_6h = predictor.process_and_predict(
                        model_path_6h, self.model_variables, raw_data_path,
                        target, past6, future=6, is_automated_run=is_auto
                    )

                    # 실제 데이터와 예측 결과 병합
                    merged_data = data[['일시', target]].copy()
                    merged_data = pd.merge(merged_data, comparison_3h, on='일시', how='outer')
                    merged_data = pd.merge(merged_data, comparison_6h, on='일시', how='outer')
                    merged_data.sort_values(by='일시', inplace=True)
                    merged_data.reset_index(drop=True, inplace=True)

                    target_label = self.get_target_label(target)
                    sheet_name = f"{target}_{target_label}"

                    # 시트에 데이터 저장
                    merged_data.to_excel(writer, sheet_name=sheet_name, index=False)

        except Exception as e_excel_write:
            msg = f"전체 예측 결과 Excel 파일 작성 중 오류 ({out_path}): {e_excel_write}"
            logging.error(msg, exc_info=True)
            if not is_auto: QMessageBox.critical(None, "Excel 저장 오류", msg)
            if is_auto and hasattr(self, '_auto_timer') and self._auto_timer.isActive():
                logging.info("Excel 저장 오류로 자동 주기 실행 중단.")
                self._auto_timer.stop()
            return  # 오류 발생 시 함수 종료

            # 완료 후 알림 및 후처리
        if is_auto:
            logging.info(f"자동모드: 전체 예측 결과 생성 완료: {out_path}")
            # ── 1) 업로드 시도 ──
            if os.path.exists(out_path):
                try:
                    success = self.upload_to_google_drive_direct_api(out_path, self.google_file_id)
                    if success:
                        logging.info(f"Google Drive 업로드 성공: {out_path}")
                    else:
                        logging.error(f"Google Drive 업로드 실패(존재는 하나 실패): {out_path}")
                except Exception as e:
                    logging.error(f"Google Drive 업로드 예외 발생: {e}", exc_info=True)
            else:
                logging.error(f"자동모드: 업로드할 파일이 존재하지 않습니다: {out_path}")
            # ── 2) 업로드 성공 여부와 관계없이 다음 주기 예약 ──
            if hasattr(self, '_auto_timer') and self._auto_timer.isActive():
                logging.info("다음 자동 실행 주기를 예약합니다.")
                self._schedule_next_cycle()

        else:
            QMessageBox.information(None, "전체 예측 완료", f"파일이 저장되었습니다: {out_path}")

    def get_model_files(self, station_code_str, future_hours):
        """지정된 관측소 코드와 예측 시간(3h 또는 6h)에 해당하는 모델 파일 경로를 반환."""
        # 모델 파일명 규칙: {관측소코드}_model_{버전/날짜}_{예측시간}h_{타입}.h5
        # 예시 딕셔너리 (실제 모델 파일명으로 업데이트 필요)
        models_for_3h = {
            '1002698': '1002698_model_2503_3h_T1.h5',
            '1006690': '1006690_model_2503_3h_T1.h5',
            '1007625': '1007625_model_2503_3h_T1.h5',
            '1014650': '1014650_model_2503_3h_T1.h5',
            '1014680': '1014680_model_2503_3h_T2.h5', #T2:past24
            '1016670': '1016670_model_2503_3h_T1.h5',
            #'1018630': '1018630_model_2503_3h_T1.h5',
            #'1018638': '1018638_model_2503_3h_T1.h5',
            #'1018675': '1018675_model_2503_3h_T1.h5',
            '1101630': '1101630_model_2503_3h_T2.h5', #T2:past24
            #'3008670': '3008670_model_2503_3h_T2.h5', #T2:past24
            #'3009680': '3009680_model_2503_3h_T2.h5', #T2:past24
            '3009693': '3009693_model_2503_3h_T1.h5',
            '3010660': '3010660_model_2503_3h_T1.h5',
            #'3011645': '3011645_model_2503_3h_T1.h5',
            '3011665': '3011665_model_2503_3h_T1.h5',
            '3011685': '3011685_model_2503_3h_T1.h5',
            #'3301654': '3301654_model_241111_3h_T3.h5',
            #'3301665': '3301665_model_241111_3h_T3.h5',
            #'3302645': '3302645_model_241111_3h_T2.h5',
            '3302658': '3302658_model_241111_3h_T2.h5',
            '4004615': '4004615_model_241111_3h_T2.h5',
            '4004660': '4004660_model_241111_3h_T2.h5',
            '4004690': '4004690_model_241111_3h_T2.h5',
            '4006660': '4006660_model_241111_3h_T2.h5',
            '4008670': '4008670_model_241111_3h_T2.h5',
            '4009610': '4009610_model_241111_3h_T2.h5',
            '4009630': '4009630_model_241111_3h_T2.h5',
            '4009640': '4009640_model_241111_3h_T2.h5',
            '5001640': '5001640_model_241111_3h_T2.h5',
            '5001645': '5001645_model_241111_3h_T2.h5',
            #'5001660': '5001660_model_241111_3h_T2.h5',
        }
        models_for_6h = {
            '1002698': '1002698_model_2503_6h_T1.h5',
            '1006690': '1006690_model_2503_6h_T1.h5',
            '1007625': '1007625_model_2503_6h_T1.h5',
            '1014650': '1014650_model_2503_6h_T1.h5',
            '1014680': '1014680_model_2503_6h_T2.h5', #T2:past24
            '1016670': '1016670_model_2503_6h_T1.h5',
            #'1018630': '1018630_model_2503_6h_T1.h5',
            #'1018638': '1018638_model_2503_6h_T1.h5',
            #'1018675': '1018675_model_2503_6h_T1.h5',
            '1101630': '1101630_model_2503_6h_T2.h5', #T2:past24
            #'3008670': '3008670_model_2503_6h_T2.h5', #T2:past24
            #'3009680': '3009680_model_2503_6h_T2.h5', #T2:past24
            '3009693': '3009693_model_2503_6h_T2.h5', #T2:past24
            '3010660': '3010660_model_2503_6h_T1.h5',
            #'3011645': '3011645_model_2503_6h_T1.h5',
            '3011665': '3011665_model_2503_6h_T1.h5',
            '3011685': '3011685_model_2503_6h_T1.h5',
            #'3301654': '3301654_model_241111_6h_T3.h5',
            #'3301665': '3301665_model_241111_6h_T3.h5',
            #'3302645': '3302645_model_241111_6h_T2.h5',
            '3302658': '3302658_model_241111_6h_T2.h5',
            '4004615': '4004615_model_241111_6h_T2.h5',
            '4004660': '4004660_model_241111_6h_T2.h5',
            '4004690': '4004690_model_241111_6h_T2.h5',
            '4006660': '4006660_model_241111_6h_T2.h5',
            '4008670': '4008670_model_241111_6h_T2.h5',
            '4009610': '4009610_model_241111_6h_T2.h5',
            '4009630': '4009630_model_241111_6h_T2.h5',
            '4009640': '4009640_model_241111_6h_T2.h5',
            '5001640': '5001640_model_241111_6h_T2.h5',
            '5001645': '5001645_model_241111_6h_T2.h5',
            #'5001660': '5001660_model_241111_6h_T2.h5',
        }

        model_filename_str = None
        if future_hours == 3:
            model_filename_str = models_for_3h.get(station_code_str)
        elif future_hours == 6:
            model_filename_str = models_for_6h.get(station_code_str)

        if not model_filename_str:
            logging.warning(f"관측소 [{station_code_str}], {future_hours}시간 예측 모델 파일명을 찾을 수 없습니다.")
            return None

        # resource_path를 사용하여 'models' 폴더 내의 파일 경로 반환
        full_model_path = resource_path(os.path.join('models', model_filename_str))
        if not os.path.exists(full_model_path):
            logging.error(f"모델 파일이 실제 경로에 존재하지 않습니다: {full_model_path}")
            return None
        return full_model_path

    def get_past_hours(self, station_code_str, future_hours: int) -> int:
        """모델 학습에 사용된 과거 데이터 시간(타임스텝 수)을 반환."""
        if future_hours == 3:
            return self.PAST_HOURS_3H.get(station_code_str, 24)  # 기본값 24시간
        elif future_hours == 6:
            return self.PAST_HOURS_6H.get(station_code_str, 24)  # 기본값 24시간
        logging.warning(f"알 수 없는 예측 시간({future_hours})에 대한 과거 데이터 시간 요청. 기본값 24 반환.")
        return 24

    def get_target_label(self, target):
        for i in range(self.comboBox.count()):
            if self.comboBox.itemText(i).startswith(target):
                return self.comboBox.itemText(i)[8:]  # 지점 코드 이후의 이름 반환
        return target  # 해당 지점을 찾지 못한 경우 지점 코드 반환

    def display_predictions(self, merged_data, target, folder_path):
        # 실제 데이터 표시 (마지막 4개)
        last_actual = merged_data[['일시', target]].dropna(subset=[target]).tail(4)
        self.tableWidget.setRowCount(len(last_actual))
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(['실제 시간', '실제 수위'])

        for i, (idx, row) in enumerate(last_actual.iterrows()):
            time_item = QTableWidgetItem(row['일시'].strftime('%Y-%m-%d %H:%M'))
            time_item.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(i, 0, time_item)
            level = row[target]
            level_item = QTableWidgetItem(f"{level:.2f}" if not pd.isna(level) else '')
            level_item.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(i, 1, level_item)

        # 예측 결과 표시 (3시간 예측 결과만 표시)
        future_predictions = merged_data[['일시', '예측 수위(3시간)']].dropna(subset=['예측 수위(3시간)']).tail(7)
        self.tableWidget_2.setRowCount(len(future_predictions))
        self.tableWidget_2.setColumnCount(2)
        self.tableWidget_2.setHorizontalHeaderLabels(['예측 시간', '예측 수위'])

        # 스타일 설정
        brush = QBrush(QColor(170, 255, 255, 100))
        brush.setStyle(Qt.SolidPattern)
        brush2 = QBrush(QColor(170, 255, 255, 150))
        brush2.setStyle(Qt.SolidPattern)
        font7 = QFont()
        font7.setFamilies([u"\ud55c\ucef4 \ub9d0\ub791\ub9d0\ub791 Regular"])
        font7.setPointSize(9)

        prev_pred_value = None
        for i, (idx, row) in enumerate(future_predictions.iterrows()):
            pred_time_item = QTableWidgetItem(row['일시'].strftime('%Y-%m-%d %H:%M'))
            pred_time_item.setTextAlignment(Qt.AlignCenter)
            pred_time_item.setFont(font7)
            self.tableWidget_2.setItem(i, 0, pred_time_item)

            pred_value = row['예측 수위(3시간)']
            # 변동 기호 결정 및 색상 설정
            if prev_pred_value is not None:
                if pred_value > prev_pred_value:
                    change_symbol = '↑'
                    symbol_color = QColor('red')
                elif pred_value < prev_pred_value:
                    change_symbol = '↓'
                    symbol_color = QColor('blue')
                else:
                    change_symbol = '-'
                    symbol_color = QColor('black')
            else:
                change_symbol = ''
                symbol_color = QColor('black')

            prev_pred_value = pred_value

            cell_widget = QWidget()
            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignCenter)
            value_label = QLabel(f"{pred_value:.2f}  ")
            value_label.setFont(font7)
            symbol_label = QLabel(change_symbol)
            symbol_label.setFont(font7)
            symbol_label.setStyleSheet(f"color: {symbol_color.name()};")
            layout.addWidget(value_label)
            layout.addWidget(symbol_label)
            cell_widget.setLayout(layout)

            self.tableWidget_2.setCellWidget(i, 1, cell_widget)

            if i >= len(future_predictions) - 4:  # 마지막 4개의 예측 값을 강조
                background_brush = brush if i % 2 == 0 else brush2
                pred_time_item.setBackground(background_brush)
                cell_widget.setStyleSheet(f"background-color: {background_brush.color().name()};")

        target_label = self.comboBox.currentText()
        self.display_graph(
            actual_times=merged_data['일시'][-78:],
            actual_levels=merged_data[target][-78:],
            predicted_times_3h=merged_data['일시'][-78:],
            predicted_levels_3h=merged_data['예측 수위(3시간)'][-78:].values,
            predicted_times_6h=merged_data['일시'][-78:],
            predicted_levels_6h=merged_data['예측 수위(6시간)'][-78:].values,
            title=f'{target_label} 지점 수위 예측(3일)',
            save_path=folder_path,
            generate_animation=False
        )
        self.display_graph(
            actual_times=merged_data['일시'][-30:],
            actual_levels=merged_data[target][-30:],
            predicted_times_3h=merged_data['일시'][-30:],
            predicted_levels_3h=merged_data['예측 수위(3시간)'][-30:].values,
            predicted_times_6h=merged_data['일시'][-30:],
            predicted_levels_6h=merged_data['예측 수위(6시간)'][-30:].values,
            title=f'{target_label} 지점 수위 예측(1일)',
            save_path=folder_path
        )

    def display_graph(self, actual_times, actual_levels, predicted_times_3h, predicted_levels_3h, predicted_times_6h,
                      predicted_levels_6h, title, save_path, generate_animation=True, display=True):
        plt.figure(figsize=(4, 2), dpi=300)
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['font.size'] = 4
        plt.rcParams['axes.titlesize'] = 4
        plt.rcParams['axes.labelsize'] = 4
        plt.rcParams['xtick.labelsize'] = 3.5
        plt.rcParams['ytick.labelsize'] = 3.5
        plt.rcParams['legend.fontsize'] = 4
        plt.plot(predicted_times_3h, predicted_levels_3h, label='예측 수위(3시간)', linestyle='-', marker='o', markersize=1,
                 linewidth=0.5, alpha=0.8, color='orange')
        plt.plot(predicted_times_6h, predicted_levels_6h, label='예측 수위(6시간)', linestyle='--', marker='^', markersize=1,
                 linewidth=0.5, alpha=0.5, color='green')
        plt.plot(actual_times, actual_levels, label='실제 수위', marker='o', markersize=1, linewidth=0.7, color='red')
        plt.xlabel('시간')
        plt.ylabel('수위(h)m')
        plt.title(title)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        ymin = min(min(actual_levels), min(predicted_levels_3h), min(predicted_levels_6h)) - 0.2
        ymax = max(max(actual_levels), max(predicted_levels_3h), max(predicted_levels_6h)) * 1.1
        y_range = ymax - ymin

        if y_range < 0.5:
            mid_point = (ymax + ymin) / 2
            ymin = mid_point - 0.25
            ymax = mid_point + 0.25

        plt.ylim(bottom=ymin, top=ymax)

        file_name = f"{title.replace(' ', '_')}.png"
        full_file_path = os.path.join(save_path, file_name)
        plt.savefig(full_file_path, format='png')
        plt.close()

        if display:
            image = QPixmap(full_file_path)

            dialog = QDialog()
            dialog.setWindowTitle(title)
            layout = QVBoxLayout()
            label = QLabel()
            label.setPixmap(image)
            layout.addWidget(label)
            dialog.setLayout(layout)
            dialog.exec()

        if generate_animation:
            # 시간 데이터를 datetime 형식으로 변환
            actual_times = pd.to_datetime(actual_times)
            predicted_times_3h = pd.to_datetime(predicted_times_3h)
            predicted_times_6h = pd.to_datetime(predicted_times_6h)

            num_actual = 30
            num_pred_3h = 30
            num_pred_6h = 30

            actual_times = actual_times[-num_actual:]
            actual_levels = actual_levels[-num_actual:]
            predicted_times_3h = predicted_times_3h[-num_pred_3h:]
            predicted_levels_3h = predicted_levels_3h[-num_pred_3h:]
            predicted_times_6h = predicted_times_6h[-num_pred_6h:]
            predicted_levels_6h = predicted_levels_6h[-num_pred_6h:]

            # 모든 데이터 시리즈의 시작 시간을 6시간 예측 데이터의 시작 시간으로 맞춤
            start_time = predicted_times_6h.iloc[0]

            # 각 데이터 시리즈의 시간 조정
            actual_time_shift = start_time - actual_times.iloc[0]
            predicted_3h_time_shift = start_time - predicted_times_3h.iloc[0]

            actual_times_adjusted = actual_times + actual_time_shift
            predicted_times_3h_adjusted = predicted_times_3h + predicted_3h_time_shift
            predicted_times_6h_adjusted = predicted_times_6h  # 이미 시작 시간이 맞춰져 있음

            # 시간 데이터를 숫자로 변환
            actual_times_num = date2num(actual_times_adjusted)
            predicted_times_3h_num = date2num(predicted_times_3h_adjusted)
            predicted_times_6h_num = date2num(predicted_times_6h_adjusted)

            # x축 범위 설정 (시작 시간에서 1시간 전부터 종료 시간에서 1시간 후까지)
            x_min = predicted_times_6h_num.min() - 1 / 24  # 시작 시간 -1시간
            x_max = predicted_times_6h_num.max() + 1 / 24  # 마지막 시간 +1시간

            # y축 범위 설정
            ymin = min(actual_levels.min(), predicted_levels_3h.min(), predicted_levels_6h.min()) - 0.1
            ymax = max(actual_levels.max(), predicted_levels_3h.max(), predicted_levels_6h.max()) + 0.1

            # 그래프 설정
            fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
            ax.set_xlabel('시간')
            ax.set_ylabel('수위(h)m')
            ax.set_title(f"{title} (애니메이션)")
            ax.grid(True, linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(ymin, ymax)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(HourLocator())  # 매 정시로 x축 눈금 설정

            # 라인 객체 생성
            line_pred_3h, = ax.plot([], [], label='예측 수위(3시간)', linestyle='-', marker='o', markersize=1, linewidth=0.5,
                                    alpha=0.8, color='orange')
            line_pred_6h, = ax.plot([], [], label='예측 수위(6시간)', linestyle='--', marker='^', markersize=1, linewidth=0.5,
                                    alpha=0.5, color='green')
            line_actual, = ax.plot([], [], label='실제 수위', marker='o', markersize=1, linewidth=0.7, color='red')

            ax.legend()

            # 각 데이터 시리즈의 시작 프레임 설정
            start_frame_6h = 0  # 6시간 예측 데이터는 프레임 0부터 표시
            start_frame_3h = 3  # 3시간 예측 데이터는 프레임 3부터 표시
            start_frame_actual = 6  # 실제 데이터는 프레임 6부터 표시

            # 총 프레임 수 설정
            total_frames = max(len(predicted_levels_6h) + start_frame_6h,
                               len(predicted_levels_3h) + start_frame_3h,
                               len(actual_levels) + start_frame_actual)

            # 애니메이션 함수 정의
            def animate(i):
                lines = []

                # 6시간 예측 데이터 업데이트
                if i >= start_frame_6h:
                    idx_6h = min(i - start_frame_6h, num_pred_6h - 1)
                    line_pred_6h.set_data(predicted_times_6h_num[:idx_6h + 1], predicted_levels_6h[:idx_6h + 1])
                    lines.append(line_pred_6h)

                # 3시간 예측 데이터 업데이트
                if i >= start_frame_3h:
                    idx_3h = min(i - start_frame_3h, num_pred_3h - 1)
                    line_pred_3h.set_data(predicted_times_3h_num[:idx_3h + 1], predicted_levels_3h[:idx_3h + 1])
                    lines.append(line_pred_3h)

                # 실제 데이터 업데이트
                if i >= start_frame_actual:
                    idx_actual = min(i - start_frame_actual, num_actual - 1)
                    line_actual.set_data(actual_times_num[:idx_actual + 1], actual_levels[:idx_actual + 1])
                    lines.append(line_actual)

                return lines

            ani = FuncAnimation(fig, animate, frames=total_frames, interval=500, blit=True)

            anim_file_name = f"{title.replace(' ', '_')}_animation.gif"
            anim_full_file_path = os.path.join(save_path, anim_file_name)
            ani.save(anim_full_file_path, writer='pillow', fps=2)

            plt.close(fig)

            if display:
                dialog_anim = QDialog()
                dialog_anim.setWindowTitle(f"{title} (애니메이션)")
                layout_anim = QVBoxLayout()
                label_anim = QLabel()
                movie = QMovie(anim_full_file_path)
                label_anim.setMovie(movie)
                movie.start()
                layout_anim.addWidget(label_anim)
                dialog_anim.setLayout(layout_anim)
                dialog_anim.exec()

# --- 메인 실행 로직 ---
def one_cycle():
    # 기존 download_clicked() → on_download_completed() → predict_all_locations() → upload…
    ui = Ui_KIHS()  # 또는 core 함수만 떼어내서 직접 호출
    ui.download_clicked()

def headless_loop(ui: Ui_KIHS):
    """헤드리스 모드: ① 켜진 즉시 download_clicked() → ② 다음 xx:10까지 대기 → ③ 매시간 xx:10에 반복"""
    # ① 프로그램 시작하자마자 한 번 실행
    try:
        logging.info("[헤드리스] 첫 실행 즉시 실행")
        ui.download_clicked()
    except Exception as e:
        logging.error(f"[헤드리스] 첫 실행 예외: {e}", exc_info=True)

    # ② 다음 xx:10 시각까지 대기
    now = datetime.now()
    next_run = now.replace(minute=10, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(hours=1)
    wait_sec = (next_run - now).total_seconds()
    logging.info(f"[헤드리스] 다음 실행까지 대기: {wait_sec/60:.1f}분 (다음 실행 시각: {next_run})")
    time.sleep(wait_sec)

    # ③ 매시간 xx:10에 반복
    while True:
        try:
            ui.download_clicked()
        except Exception as e:
            logging.error(f"[헤드리스] download_clicked() 예외: {e}", exc_info=True)
        # 다음 실행까지 대기 (1시간)
        logging.info(f"[헤드리스] 다음 실행까지 1시간 대기")
        time.sleep(3600)

def main(): # main 함수 정의는 그대로 유지
    # 헤드리스 vs GUI 분기
    if '--auto' in sys.argv:
        app = QCoreApplication(sys.argv)
        ui = Ui_KIHS()
        ui.is_headless = True
        ui.setupUi(None)  # None 을 넘겨서 GUI 요소 없이 초기화
        ui.checkBox_auto.setChecked(True)
        logging.info("헤드리스 자동주기 모드 시작")
        # 다운로드→예측→업로드를 트리거하는 타이머 준비
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(ui.download_clicked)
        # 다음 실행 시간 계산 함수
        def schedule_next():
            now = datetime.now()
            # 매시 xx:10 에 실행
            next_run = now.replace(minute=10, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(hours=1)
            wait_ms = int((next_run - now).total_seconds() * 1000)
            logging.info(f"[헤드리스] 다음 실행까지 대기: {wait_ms / 1000 / 60:.1f}분 (실행 시각: {next_run})")
            timer.start(wait_ms)

        # ① 첫 실행 즉시
        ui.download_clicked()
        # ② 다운로드 완료 신호가 올 때마다 다음 스케줄 등록
        ui.download_thread_instance.download_completed.connect(lambda _: schedule_next())
        # ③ 첫 실행 후 다음 사이클 예약
        schedule_next()
        # 이벤트 루프 시작
        return app.exec()
    else:
        # 기존 GUI 모드
        app = QApplication(sys.argv)
        dlg = QDialog()
        ui = Ui_KIHS()
        ui.setupUi(dlg)
        dlg.show()
        return app.exec()

if __name__ == '__main__':
    sys.exit(main())
