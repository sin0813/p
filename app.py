
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import matplotlib.font_manager as fm

st.set_page_config(
    page_title="인공지능 사진 구분",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 인공지능 사진 구분")
st.write("주파수 분석을 이용해 AI 생성 사진과 실제 사진의 특징을 비교합니다.")

# 한글 그래프 글꼴 설정
plt.rcParams["axes.unicode_minus"] = False

font_candidates = [
    "Malgun Gothic",
    "NanumGothic",
    "AppleGothic"
]

available_fonts = {f.name for f in fm.fontManager.ttflist}

for font in font_candidates:
    if font in available_fonts:
        plt.rcParams["font.family"] = font
        break


# 사진의 주파수 특징 계산
def spectrum(image, bins=64):
    image = image.convert("L").resize((256, 256))
    arr = np.asarray(image, dtype=np.float32)

    fft = np.fft.fftshift(np.fft.fft2(arr))
    power = np.abs(fft) ** 2

    y, x = np.indices(power.shape)
    center = 128
    radius = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    max_radius = 128
    result = []

    for i in range(bins):
        mask = (
            (radius >= i * max_radius / bins) &
            (radius < (i + 1) * max_radius / bins)
        )

        values = power[mask]
        result.append(np.mean(values) if len(values) else 0)

    result = np.array(result)
    result = np.log1p(result)

    if result.max() > result.min():
        result = (result - result.min()) / (
            result.max() - result.min()
        )

    return result


# 예시 사진 불러오기
BASE_DIR = Path(__file__).resolve().parent

possible_paths = [
    BASE_DIR / "sample_images",
    BASE_DIR / "ai_photo_detector_project" / "sample_images",
    Path("sample_images"),
]

SAMPLES = next(
    (p for p in possible_paths if p.exists() and p.is_dir()),
    BASE_DIR / "sample_images"
)


def read_examples(prefix):
    profiles = []
    loaded_files = []

    if not SAMPLES.exists():
        return profiles, loaded_files

    for path in sorted(SAMPLES.iterdir()):
        if path.is_file() and path.suffix.lower() in [
            ".jpg", ".jpeg", ".png", ".webp"
        ]:
            if path.name.lower().startswith(prefix):
                try:
                    image = Image.open(path).convert("RGB")
                    profiles.append(spectrum(image))
                    loaded_files.append(path.name)
                except Exception as e:
                    st.warning(f"{path.name} 읽기 실패: {e}")

    return profiles, loaded_files


ai_profiles, ai_files = read_examples("ai")
real_profiles, real_files = read_examples("real")
st.write("현재 실행 파일:", Path(__file__).resolve())
st.write("현재 폴더:", Path(__file__).resolve().parent)
st.write("예시 사진 폴더:", SAMPLES.resolve())
st.write("예시 폴더 존재 여부:", SAMPLES.exists())

if SAMPLES.exists():
    st.write("폴더 안 파일:", [
        p.name for p in SAMPLES.iterdir()
    ])


# 예시 데이터 상태
with st.expander("예시 사진 불러오기 상태 확인"):
    st.write(f"AI 사진: {len(ai_profiles)}장")
    st.write(f"실제 사진: {len(real_profiles)}장")

    if ai_files:
        st.write("AI 사진 목록:", ", ".join(ai_files))

    if real_files:
        st.write("실제 사진 목록:", ", ".join(real_files))

    if not ai_profiles or not real_profiles:
        st.error(
            "예시 사진을 불러오지 못했습니다. "
            "sample_images 폴더와 파일 이름을 확인해 주세요."
        )


# 예시 사진 평균 그래프
if ai_profiles and real_profiles:
    ai_average = np.mean(ai_profiles, axis=0)
    real_average = np.mean(real_profiles, axis=0)

    st.subheader("AI 사진과 실제 사진의 평균 주파수 비교")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ai_average, label="AI 사진 평균")
    ax.plot(real_average, label="실제 사진 평균")
    ax.set_xlabel("주파수 구간")
    ax.set_ylabel("정규화된 주파수 특징")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# 사진 업로드
st.divider()
st.subheader("사진 분석하기")

uploaded_file = st.file_uploader(
    "분석할 사진을 업로드하세요.",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="업로드한 사진",
        use_container_width=True
    )

    uploaded_profile = spectrum(image)

    st.subheader("업로드 사진의 주파수 그래프")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(uploaded_profile, label="업로드 사진")
    ax.set_xlabel("주파수 구간")
    ax.set_ylabel("정규화된 주파수 특징")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # AI와 실제 사진 비교
    if ai_profiles and real_profiles:
        ai_average = np.mean(ai_profiles, axis=0)
        real_average = np.mean(real_profiles, axis=0)

        ai_distance = np.mean(
            (uploaded_profile - ai_average) ** 2
        )

        real_distance = np.mean(
            (uploaded_profile - real_average) ** 2
        )

        # 유사도 점수 계산
        temperature = 0.02

        scores = np.exp(
            -np.array([ai_distance, real_distance]) / temperature
        )

        percentages = scores / scores.sum() * 100

        ai_percent = percentages[0]
        real_percent = percentages[1]

        st.divider()
        st.subheader("사진 분석 결과")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("AI 사진 유사도", f"{ai_percent:.1f}%")

        with col2:
            st.metric("실제 사진 유사도", f"{real_percent:.1f}%")

        if ai_percent > real_percent:
            st.info("AI 사진 예시와 주파수 특징이 더 비슷합니다.")
        else:
            st.info("실제 사진 예시와 주파수 특징이 더 비슷합니다.")

        st.caption(
            "주의: 이 수치는 실제 AI 생성 확률이 아니라 "
            "현재 등록된 예시 사진과의 상대적인 유사도입니다. "
            "주파수 특징만으로 사진의 진위를 확정할 수 없습니다."
        )

    else:
        st.error(
            "예시 사진이 부족하여 판별 점수를 계산할 수 없습니다. "
            "AI 사진과 실제 사진이 각각 정상적으로 불러와지는지 "
            "위의 상태 확인 메뉴에서 확인해 주세요."
        )
