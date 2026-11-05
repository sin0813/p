import io
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(page_title="인공지능 사진 구분", page_icon="🔎", layout="wide")
ROOT = Path(__file__).parent
SAMPLES = ROOT / "sample_images"
MAX_BYTES = 15 * 1024 * 1024

st.markdown("""
<style>
.big-title{font-size:2.6rem;font-weight:850;letter-spacing:-1.2px}
.small-note{color:#64748b;font-size:.92rem}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="big-title">인공지능 사진 구분</div>', unsafe_allow_html=True)
st.write("사진의 주파수 특징을 그래프로 살펴보고, 제공된 예시 사진과 비교하는 교육용 실험 사이트입니다.")
st.warning("주의: 주파수 그래프만으로 AI 사진인지 실제 사진인지 확실하게 판별할 수는 없습니다. 아래 비교 결과는 소수의 예시 사진에 기반한 참고용이며, 정확한 AI 탐지기가 아닙니다.")

def spectrum(image, bins=64):
    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32) / 255.0
    gray -= gray.mean()
    window = np.outer(np.hanning(256), np.hanning(256))
    fft = np.fft.fftshift(np.fft.fft2(gray * window))
    power = np.log1p(np.abs(fft) ** 2)
    yy, xx = np.indices(power.shape)
    radius = np.sqrt((yy - 127.5)**2 + (xx - 127.5)**2)
    radius /= radius.max()
    edges = np.linspace(0, 1, bins + 1)
    values = []
    for i in range(bins):
        mask = (radius >= edges[i]) & (radius < edges[i+1])
        values.append(float(power[mask].mean()) if mask.any() else 0.0)
    values = np.asarray(values, dtype=np.float32)
    values -= values.min()
    if values.max() > 1e-9:
        values /= values.max()
    return values

def read_examples(prefix):
    profiles = []
    if SAMPLES.exists():
        for path in sorted(SAMPLES.glob(prefix + "_*")):
            try:
                with Image.open(path) as im:
                    profiles.append(spectrum(im.convert("RGB")))
            except Exception:
                pass
    return profiles

ai_profiles = read_examples("ai")
real_profiles = read_examples("real")

st.header("1. 사진은 왜 주파수로 살펴볼까?")
st.write("""
사진은 작은 점인 픽셀들이 모여 만들어집니다. 넓은 하늘이나 벽처럼 색과 밝기가 천천히 변하는 부분은
**낮은 주파수**, 머리카락·털·글자처럼 밝기가 빠르게 바뀌는 세밀한 부분은 **높은 주파수**와 관련이 있습니다.
푸리에 변환(FFT)은 사진을 이런 주파수 성분으로 나누어 살펴보는 수학적 방법입니다.
""")
st.info("""
쉽게 말해 낮은 주파수는 사진의 큰 명암 덩어리, 높은 주파수는 작은 무늬와 경계선에 가깝습니다.
하지만 카메라, 사진 내용, 보정, 압축, 크기 변경에 따라서도 주파수 그래프가 달라집니다.
따라서 'AI 사진은 항상 이런 모양'이라고 정해진 패턴은 없습니다.
""")

st.header("2. 예시 사진의 주파수 모양 비교")
if ai_profiles and real_profiles:
    ai_avg = np.mean(ai_profiles, axis=0)
    real_avg = np.mean(real_profiles, axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(real_avg, label=f"실제 사진 예시 (n={len(real_profiles)})", linewidth=2)
    ax.plot(ai_avg, label=f"AI 사진 예시 (n={len(ai_profiles)})", linewidth=2)
    ax.set_xlabel("주파수 구간  (왼쪽: 낮음 → 오른쪽: 높음)")
    ax.set_ylabel("정규화된 상대 에너지")
    ax.set_title("제공된 예시 사진들의 평균 주파수 프로필")
    ax.grid(alpha=.25)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("이 그래프는 이번 프로젝트에 넣은 예시 사진들의 평균이며, 모든 AI/실제 사진을 대표하지 않습니다.")
else:
    st.info("sample_images 폴더에 ai_*.jpg와 real_*.jpg 파일을 넣으면 비교 그래프가 표시됩니다.")

st.header("3. 사진 업로드 및 실시간 주파수 분석")
uploaded = st.file_uploader("사진 파일을 선택하세요 (JPG, JPEG, PNG, WEBP / 최대 15MB)", type=["jpg","jpeg","png","webp"])
if uploaded is not None:
    try:
        raw = uploaded.getvalue()
        if not raw:
            raise ValueError("파일이 비어 있습니다.")
        if len(raw) > MAX_BYTES:
            raise ValueError("파일 크기가 15MB를 넘습니다.")
        try:
            im = Image.open(io.BytesIO(raw))
            if im.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("지원하지 않는 이미지 형식입니다.")
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise ValueError("이미지를 읽을 수 없습니다. 정상적인 JPG/PNG/WEBP 파일인지 확인해 주세요.")
        st.image(im, caption="업로드한 사진", use_container_width=True)
        profile = spectrum(im)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(profile, linewidth=2, label="업로드한 사진")
        ax.set_xlabel("주파수 구간  (왼쪽: 낮음 → 오른쪽: 높음)")
        ax.set_ylabel("정규화된 상대 에너지")
        ax.set_ylim(0, 1.05)
        ax.set_title("업로드한 사진의 주파수 그래프")
        ax.grid(alpha=.25)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if ai_profiles and real_profiles:
            ai_d = float(np.mean((profile - np.mean(ai_profiles, axis=0))**2))
            real_d = float(np.mean((profile - np.mean(real_profiles, axis=0))**2))
            c1, c2 = st.columns(2)
            c1.metric("AI 예시 평균과의 거리", f"{ai_d:.4f}")
            c2.metric("실제 사진 예시 평균과의 거리", f"{real_d:.4f}")
            if ai_d < real_d:
                st.warning("주파수 모양만 보면 현재 AI 예시 그룹에 조금 더 가깝습니다.")
            elif real_d < ai_d:
                st.success("주파수 모양만 보면 현재 실제 사진 예시 그룹에 조금 더 가깝습니다.")
            else:
                st.info("두 예시 그룹과의 거리가 같습니다.")
            st.caption("거리 값은 확률(%)이 아닙니다. 예시가 AI 6장, 실제 7장뿐이므로 결과는 참고용입니다.")
        else:
            st.info("예시 데이터가 없어 그룹 비교는 생략했습니다. 주파수 그래프는 표시했습니다.")
    except ValueError as e:
        st.error(f"파일 처리 오류: {e}")
    except MemoryError:
        st.error("이미지가 너무 커 메모리가 부족합니다. 크기를 줄여 다시 시도해 주세요.")
    except Exception:
        st.error("예상하지 못한 오류가 발생했습니다. 다른 이미지 파일로 다시 시도해 주세요.")
else:
    st.caption("사진을 업로드하면 이곳에 그래프와 비교 결과가 표시됩니다.")

st.header("4. 판별 결과를 볼 때 주의할 점")
st.markdown("""
- 주파수 그래프는 단서이지 증거가 아닙니다.
- 실제 사진도 강한 보정이나 압축을 거치면 AI 사진과 비슷한 그래프가 나올 수 있습니다.
- AI 생성 기술은 계속 발전하므로 특정 패턴 하나로 모든 AI 사진을 찾아낼 수 없습니다.
- 실제 탐지기를 만들려면 여러 카메라와 AI 모델, 다양한 장면에서 수천 장 이상의 학습·검증 데이터가 필요합니다.
- 사진 출처, 촬영 정보(EXIF), 원본 게시물 등 다른 정보도 함께 확인하는 것이 좋습니다.
""")
st.caption("고등학생 수준의 FFT 체험용 프로젝트 · 중요한 판단의 유일한 근거로 사용하지 마세요.")
