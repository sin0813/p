
import io
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, UnidentifiedImageError

# 기본 설정
st.set_page_config(
    page_title="인공지능 사진 구분",
    page_icon="🔎",
    layout="wide"
)

ROOT = Path(__file__).parent
SAMPLES = ROOT / "sample_images"
MAX_BYTES = 15 * 1024 * 1024

# 제목 디자인
st.markdown("""
<style>
.big-title {
    font-size: 2.6rem;
    font-weight: 850;
    letter-spacing: -1.2px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="big-title">인공지능 사진 구분</div>',
    unsafe_allow_html=True
)

st.write(
    "사진의 주파수 특징을 분석하여 "
    "AI 사진인지 실제 사진인지 비교해 보는 사이트입니다."
)

st.warning(
    "주의: 주파수 분석만으로 AI 사진 여부를 "
    "100% 정확하게 판별할 수는 없습니다."
)


# 사진의 주파수 분석 함수
def spectrum(image, bins=64):
    gray = np.asarray(
        image.convert("L").resize((256, 256)),
        dtype=np.float32
    ) / 255.0

    gray -= gray.mean()

    window = np.outer(
        np.hanning(256),
        np.hanning(256)
    )

    fft = np.fft.fftshift(
        np.fft.fft2(gray * window)
    )

    power = np.log1p(np.abs(fft) ** 2)

    yy, xx = np.indices(power.shape)

    radius = np.sqrt(
        (yy - 127.5) ** 2 +
        (xx - 127.5) ** 2
    )

    radius /= radius.max()

    edges = np.linspace(0, 1, bins + 1)

    values = []

    for i in range(bins):
        mask = (
            (radius >= edges[i]) &
            (radius < edges[i + 1])
        )

        if mask.any():
            values.append(float(power[mask].mean()))
        else:
            values.append(0.0)

    values = np.asarray(values, dtype=np.float32)

    values -= values.min()

    if values.max() > 1e-9:
        values /= values.max()

    return values


# 예시 사진 불러오기
def read_examples(prefix):
    profiles = []

    if SAMPLES.exists():
        for path in sorted(SAMPLES.glob(prefix + "_*")):
            try:
                with Image.open(path) as im:
                    profiles.append(
                        spectrum(im.convert("RGB"))
                    )
            except Exception:
                pass

    return profiles


ai_profiles = read_examples("ai")
real_profiles = read_examples("real")


# 1. 주파수 설명
st.header("1. 사진은 왜 주파수로 구별할까?")

st.write("""
사진은 작은 점인 픽셀들이 모여 만들어집니다.

넓은 하늘이나 벽처럼 색과 밝기가 천천히 변하는 부분은
낮은 주파수와 관련이 있습니다.

반대로 머리카락, 털, 글자처럼 밝기가 빠르게 바뀌는
세밀한 부분은 높은 주파수와 관련이 있습니다.

푸리에 변환(FFT)은 사진을 이런 주파수 성분으로
나누어 살펴보는 수학적 방법입니다.
""")

st.info("""
쉽게 말해 낮은 주파수는 사진의 큰 명암 덩어리,
높은 주파수는 작은 무늬와 경계선에 가깝습니다.

하지만 카메라, 사진 내용, 보정, 압축에 따라서도
주파수 그래프가 달라질 수 있습니다.
""")


# 2. 예시 사진 주파수 비교
st.header("2. 예시 사진의 주파수 모양 비교")

if ai_profiles and real_profiles:

    ai_avg = np.mean(ai_profiles, axis=0)
    real_avg = np.mean(real_profiles, axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        real_avg,
        label="실제 사진",
        linewidth=2
    )

    ax.plot(
        ai_avg,
        label="AI 사진",
        linewidth=2
    )

    ax.set_xlabel("주파수 구간 (낮음 → 높음)")
    ax.set_ylabel("정규화된 상대 에너지")
    ax.set_title("AI 사진과 실제 사진의 평균 주파수 비교")

    ax.grid(alpha=0.25)
    ax.legend()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "이 그래프는 제공된 예시 사진들의 평균입니다."
    )

else:
    st.info(
        "sample_images 폴더에 AI 사진과 실제 사진을 넣어주세요."
    )


# 3. 사진 업로드
st.header("3. 사진 업로드 및 주파수 분석")

uploaded = st.file_uploader(
    "사진을 업로드하세요.",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:

    try:
        raw = uploaded.getvalue()

        if not raw:
            raise ValueError("파일이 비어 있습니다.")

        if len(raw) > MAX_BYTES:
            raise ValueError("파일 크기가 15MB를 초과했습니다.")

        try:
            im = Image.open(io.BytesIO(raw))

            if im.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("지원하지 않는 이미지입니다.")

            im = ImageOps.exif_transpose(im).convert("RGB")
            im.load()

        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError
        ):
            raise ValueError(
                "이미지를 읽을 수 없습니다. "
                "정상적인 JPG, PNG, WEBP 파일인지 확인해 주세요."
            )

        st.image(
            im,
            caption="업로드한 사진",
            use_container_width=True
        )

        # 업로드 사진 주파수 분석
        profile = spectrum(im)

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            profile,
            linewidth=2,
            label="업로드한 사진"
        )

        ax.set_xlabel("주파수 구간 (낮음 → 높음)")
        ax.set_ylabel("정규화된 상대 에너지")
        ax.set_ylim(0, 1.05)
        ax.set_title("업로드한 사진의 주파수 그래프")

        ax.grid(alpha=0.25)
        ax.legend()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # 4. AI / 실제 사진 점수 비교
        if ai_profiles and real_profiles:

            ai_avg = np.mean(ai_profiles, axis=0)
            real_avg = np.mean(real_profiles, axis=0)

            ai_d = float(
                np.mean((profile - ai_avg) ** 2)
            )

            real_d = float(
                np.mean((profile - real_avg) ** 2)
            )

            st.subheader("4. 사진 판별 결과")

            c1, c2 = st.columns(2)

            c1.metric(
                "AI 사진 예시와의 거리",
                f"{ai_d:.4f}"
            )

            c2.metric(
                "실제 사진 예시와의 거리",
                f"{real_d:.4f}"
            )

            # 거리를 상대 점수로 변환
            # 실제 검증 확률이 아닌 참고용 점수
            temperature = 0.02

            logits = np.array(
                [-ai_d, -real_d],
                dtype=np.float64
            ) / temperature

            logits -= logits.max()

            scores = np.exp(logits)
            scores /= scores.sum()

            ai_percent = float(scores[0] * 100)
            real_percent = float(scores[1] * 100)

            c1, c2 = st.columns(2)

            c1.metric(
                "AI 사진일 가능성 점수",
                f"{ai_percent:.1f}%"
            )

            c2.metric(
                "실제 사진일 가능성 점수",
                f"{real_percent:.1f}%"
            )

            st.progress(int(round(ai_percent)))

            if ai_percent >= 50:

                st.warning(
                    f"AI 사진 예시와 더 비슷합니다. "
                    f"AI 점수: {ai_percent:.1f}%"
                )

            else:

                st.success(
                    f"실제 사진 예시와 더 비슷합니다. "
                    f"실제 사진 점수: {real_percent:.1f}%"
                )

            st.caption(
                "주의: 이 퍼센트는 검증된 실제 확률이 아니라 "
                "AI 예시 6장과 실제 사진 예시 7장의 "
                "평균 주파수와 비교한 상대 점수입니다."
            )

        else:

            st.info(
                "예시 사진이 부족하여 판별 점수를 계산할 수 없습니다."
            )

    except ValueError as e:

        st.error(f"파일 처리 오류: {e}")

    except MemoryError:

        st.error(
            "이미지가 너무 커서 메모리가 부족합니다. "
            "사진 크기를 줄여 다시 시도해 주세요."
        )

    except Exception:

        st.error(
            "예상하지 못한 오류가 발생했습니다. "
            "다른 사진으로 다시 시도해 주세요."
        )

else:

    st.caption(
        "사진을 업로드하면 주파수 그래프와 판별 결과가 표시됩니다."
    )


# 5. 판별 시 주의사항
st.header("5. 판별 결과를 볼 때 주의할 점")

st.markdown("""
- 주파수 그래프는 AI 사진을 구분하는 하나의 단서입니다.
- 실제 사진도 보정이나 압축에 따라 AI 사진과 비슷해질 수 있습니다.
- 모든 AI 사진이 같은 주파수 특징을 가지는 것은 아닙니다.
- 더 정확한 판별을 위해서는 다양한 사진으로 학습한 모델이 필요합니다.
- 사진 출처와 촬영 정보도 함께 확인하는 것이 좋습니다.
""")

st.caption(
    "고등학생 수준의 FFT 체험용 프로젝트입니다. "
    "중요한 판단의 유일한 근거로 사용하지 마세요."
)
