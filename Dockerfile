FROM python:3.9-slim AS base

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ① tzdata 설치
RUN apt-get update && apt-get install -y tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata

# ② TZ 환경변수 설정
ENV TZ=Asia/Seoul

COPY . .

ENV PYTHONUNBUFFERED=1 \
    CLOUD_ENV=true \
    WORKDIR=/app

RUN mkdir -p /app/prediction_temp_files && chmod 777 /app/prediction_temp_files

COPY WORKDIR/models /tmp/models
COPY WORKDIR/scalers /tmp/scalers

CMD ["python", "AI_prediction_V250519_V8.py", "--auto"]
