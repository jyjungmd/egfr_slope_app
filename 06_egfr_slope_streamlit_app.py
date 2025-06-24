# Streamlit 웹 예측 도구 (eGFR slope 예측)
### !pip install streamlit
import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Note: If you encounter a ModuleNotFoundError for 'streamlit',
# ensure you have the correct runtime environment selected or try restarting the runtime
# after installing streamlit.

# 모델 import

st.set_page_config(page_title="eGFR Slope Prediction", layout="centered")
st.title("eGFR Slope 예측 도구 (KNOW-CKD 기반)")
st.markdown("환자의 기초 정보를 입력하면 연간 eGFR 감소 속도를 예측합니다.")

# 사용자 입력
age = st.number_input("나이 (Age)", min_value=18, max_value=100, value=65)
sex = st.selectbox("성별 (Sex)", options=["남", "여"])
bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, value=23.5)
smoking = st.selectbox("흡연 여부", options=["비흡연", "과거흡연", "현재흡연"])
dm = st.selectbox("당뇨병 여부", options=["없음", "있음"])
sbp = st.number_input("수축기 혈압 (SBP)", min_value=80, max_value=200, value=120)
egfr = st.number_input("기저 eGFR", min_value=5.0, max_value=150.0, value=55.0)
alb = st.number_input("Albumin", min_value=1.0, max_value=6.0, value=4.0)
hb = st.number_input("Hemoglobin", min_value=5.0, max_value=20.0, value=12.5)
uacr = st.number_input("UACR", min_value=0.0, max_value=3000.0, value=120.0)
upcr = st.number_input("UPCR", min_value=0.0, max_value=10.0, value=1.2)

# 입력값 정리
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [1 if sex == "남" else 0],
    'BMI': [bmi],
    'Smoking': [0 if smoking == "비흡연" else (1 if smoking == "과거흡연" else 2)],
    'DM': [1 if dm == "있음" else 0],
    'SBP': [sbp],
    'eGFR': [egfr],
    'Alb': [alb],
    'Hb': [hb],
    'UACR': [uacr],
    'UPCR': [upcr]
})

# feature 순서 일치 및 타입 통일
feature_names = ['Age', 'Sex', 'BMI', 'Smoking', 'DM', 'SBP', 'eGFR', 'Alb', 'Hb', 'UACR', 'UPCR']
input_data = input_data[feature_names]
input_data = input_data.astype(float)


# 모델 로딩
### from google.colab import files
### uploaded = files.upload()

model = joblib.load("LGBM_egfr_slope_model.pkl")



if st.button("예측하기"):
    pred = model.predict(input_data)[0]
    st.success(f"예측된 eGFR 연간 감소 속도: {pred:.2f} mL/min/1.73㎡/year")

    # SHAP 해석
    st.markdown("---")
    st.subheader("📊 변수별 영향도 (SHAP)")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches="tight")
