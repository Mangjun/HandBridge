#!/bin/bash

# 가상환경이 있는지 확인하고 없으면 생성
if [ ! -d "venv" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
source venv/bin/activate

# 의존성 설치
echo "의존성 설치 중..."
pip install -r requirements.txt

# 서버 실행
echo "서버 시작 중..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 