# 🤚 HandBridge
동양미래대 인공지능 팀 프로젝트 실시간 수어 번역 앱 이음입니다. 

## 📋 Introduce
* 🗣️ 휴대폰으로 수어 영상을 촬영하면 음성으로 번역됩니다.
* 😃 감정에 기반해 문장을 생성합니다.

## ⚙️ Environment
* 🛠️ 개발 환경
  * Client: React Native
  * Server: FastAPI
  * Model: YOLO(hand, face), nateraw/ferplus-emo, LSTM, t5 
  * Deployment: Docker
* 📅 개발 기간
  * Total: 2025.04.25 ~ 
  * Architecture: 2025.04.25 ~ 2025.05.16
  * AI: 2025.05.18 ~ 
  * Client:

## 📈 Model
* 🤚 손 탐지 모델: 이미 사전학습된 모델 사용(YOLO hand)
* 🦲 얼굴 탐지 모델: 이미 사전학습된 모델 사용(YOLO face)
* 😃 감정 추출 모델: 이미 사전학습된 모델 사용(nateraw/ferplus-emo)
* 👐 수어 단어 추출 모델: LSTM 모델 사용
* 🔠 문장 생성 모델: 이미 사전학습된 모델 사용(t5)

## ➡️ Pipeline
```
1. 수어 영상 촬영
2. 30 프레임으로 분할
3. 왼손과 오른손 keypoint 추출(YOLO hand)
4. 마지막 프레임에서 얼굴 감정 추출(YOLO face)
5. keypoint 배열과 얼굴 감정 서버로 전송(RestAPI)
6. keypoint -> npz 전처리
7. 프레임별 수어 단어 추출(LSTM)
8. 단어 배열과 얼굴 감정을 통해 문장 생성
9. 클라이언트로 전송 후 tts로 음성 출력
```

## 📁 Data
* 출처: [AI hub 수어 영상](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=103)