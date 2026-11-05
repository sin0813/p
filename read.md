# 인공지능 사진 구분

FFT(푸리에 변환) 주파수 그래프를 이용해 사진을 시각화하는 Streamlit 교육용 프로젝트입니다.

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub / Streamlit 배포
1. 이 폴더의 `app.py`, `requirements.txt`, `README.md`, `sample_images/`를 GitHub 저장소에 업로드합니다.
2. Streamlit Community Cloud에서 저장소를 연결합니다.
3. Main file path에 `app.py`를 입력하고 Deploy를 누릅니다.

## 예시 데이터와 한계
`sample_images/ai_*.jpg` 6장은 사용자가 AI 생성 이미지라고 지정한 사진이고, `real_*.jpg` 7장은 사용자가 실제 촬영 사진이라고 지정한 사진입니다. 공개 저장소에 올리기 전 실제 사진의 사용 허가와 라이선스를 확인하세요.

이 앱은 확정적인 AI 탐지 모델이 아닙니다. 업로드 사진의 FFT 주파수 프로필을 예시 그룹 평균과 비교하는 단순 시연입니다. 데이터가 13장뿐이고 장면·카메라·압축 차이가 통제되지 않았습니다. 거리 값은 확률이 아니며 AI 탐지 성능을 뜻하지 않습니다.
