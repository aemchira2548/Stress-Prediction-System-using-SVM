from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "svm_stress_model.pkl"


st.set_page_config(
    page_title="Stress Prediction SVM",
    page_icon="🧠",
    layout="centered",
)


st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .result-stress {
        padding: 20px;
        border-radius: 12px;
        background-color: #ffe5e5;
        border: 1px solid #ff8a8a;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
    }

    .result-normal {
        padding: 20px;
        border-radius: 12px;
        background-color: #e7f8ea;
        border: 1px solid #7acb86;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None

    return joblib.load(MODEL_FILE)


model = load_model()


st.markdown(
    '<div class="main-title">🧠 ระบบทำนายระดับความเครียด</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">จำแนกข้อมูลด้วย Support Vector Machine (SVM)</div>',
    unsafe_allow_html=True,
)


if model is None:
    st.error(
        "ไม่พบไฟล์ svm_stress_model.pkl "
        "กรุณารัน train_model.py ก่อน"
    )
    st.stop()


with st.form("stress_form"):
    student_type = st.selectbox(
        "ประเภทนักเรียน/นักศึกษา",
        options=["school", "college"],
        format_func=lambda value: (
            "นักเรียนมัธยม" if value == "school" else "นักศึกษา"
        ),
    )

    sleep_hours = st.number_input(
        "จำนวนชั่วโมงนอนต่อวัน",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5,
    )

    study_hours = st.number_input(
        "จำนวนชั่วโมงอ่านหนังสือต่อวัน",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5,
    )

    social_media_hours = st.number_input(
        "จำนวนชั่วโมงใช้สื่อสังคมออนไลน์ต่อวัน",
        min_value=0.0,
        max_value=24.0,
        value=3.0,
        step=0.5,
    )

    attendance = st.slider(
        "เปอร์เซ็นต์การเข้าเรียน",
        min_value=0,
        max_value=100,
        value=80,
    )

    exam_pressure = st.slider(
        "ระดับความกดดันจากการสอบ",
        min_value=0,
        max_value=10,
        value=5,
    )

    family_support = st.slider(
        "ระดับการสนับสนุนจากครอบครัว",
        min_value=0,
        max_value=10,
        value=5,
    )

    month = st.selectbox(
        "เดือน",
        options=list(range(1, 13)),
        format_func=lambda value: f"เดือนที่ {value}",
    )

    submitted = st.form_submit_button(
        "🔍 ทำนายระดับความเครียด",
        use_container_width=True,
    )


if submitted:
    new_data = pd.DataFrame(
        {
            "Student_Type": [student_type],
            "Sleep_Hours": [sleep_hours],
            "Study_Hours": [study_hours],
            "Social_Media_Hours": [social_media_hours],
            "Attendance": [attendance],
            "Exam_Pressure": [exam_pressure],
            "Family_Support": [family_support],
            "Month": [month],
        }
    )

    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]

    no_stress_probability = probabilities[0] * 100
    stress_probability = probabilities[1] * 100

    st.divider()
    st.subheader("ผลการทำนาย")

    if prediction == 1:
        st.markdown(
            '<div class="result-stress">'
            "⚠️ มีแนวโน้มเกิดความเครียด"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-normal">'
            "✅ ไม่มีแนวโน้มเกิดความเครียด"
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.write(
        f"โอกาสไม่มีความเครียด: "
        f"**{no_stress_probability:.2f}%**"
    )
    st.progress(no_stress_probability / 100)

    st.write(
        f"โอกาสมีความเครียด: "
        f"**{stress_probability:.2f}%**"
    )
    st.progress(stress_probability / 100)

    st.caption(
        "ผลลัพธ์นี้จัดทำขึ้นเพื่อการศึกษา "
        "ไม่ใช่การวินิจฉัยทางการแพทย์"
    )