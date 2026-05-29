import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('API_URL', 'http://localhost:8000')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title='Disease Prediction', layout='centered')
st.title('Disease Prediction')
st.markdown('Fill in the patient details and click **Predict**.')

# ── Sidebar — model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.header('Model Info')
    try:
        response = requests.get(f'{API_URL}/model/info', timeout=5)
        response.raise_for_status()
        info = response.json()
        st.metric('Test Accuracy',  f"{info['test_accuracy']*100:.1f}%")
        st.metric('ROC-AUC',        f"{info['test_roc_auc']:.4f}")
        st.metric('Precision',      f"{info['test_precision']:.4f}")
        st.metric('Recall',         f"{info['test_recall']:.4f}")
        st.metric('F1',             f"{info['test_f1']:.4f}")
        st.caption(f"Trained at: {info['trained_at']}")
    except Exception as e:
        st.warning(f'API unavailable: {e}')

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader('Vitals')
    age          = st.number_input('Age',                  min_value=1,   max_value=120, value=45)
    bmi          = st.number_input('BMI',                  min_value=1.0, max_value=80.0, value=25.0, step=0.1)
    glucose      = st.number_input('Glucose (mg/dL)',      min_value=1.0, max_value=600.0, value=90.0)
    cholesterol  = st.number_input('Cholesterol (mg/dL)',  min_value=1.0, max_value=600.0, value=200.0)
    systolic_bp  = st.number_input('Systolic BP',          min_value=1.0, max_value=300.0, value=120.0)
    diastolic_bp = st.number_input('Diastolic BP',         min_value=1.0, max_value=200.0, value=80.0)
    heart_rate   = st.number_input('Heart Rate',           min_value=1.0, max_value=250.0, value=70.0)

with col2:
    st.subheader('Lifestyle & History')
    gender            = st.selectbox('Gender',              ['Male', 'Female'])
    smoking           = st.selectbox('Smoking',             ['No', 'Yes'])
    alcohol           = st.selectbox('Alcohol Consumption', ['No', 'Yes'])
    physical_activity = st.selectbox('Physical Activity',   ['Low', 'Medium', 'High'])
    family_history    = st.selectbox('Family History',      ['No', 'Yes'])

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()

if st.button('Predict', use_container_width=True, type='primary'):
    payload = {
        'age':                 age,
        'bmi':                 bmi,
        'glucose_mg_dl':       glucose,
        'cholesterol_mg_dl':   cholesterol,
        'systolic_bp':         systolic_bp,
        'diastolic_bp':        diastolic_bp,
        'heart_rate':          heart_rate,
        'gender':              gender,
        'smoking':             smoking,
        'alcohol_consumption': alcohol,
        'physical_activity':   physical_activity,
        'family_history':      family_history,
    }

    try:
        response = requests.post(f'{API_URL}/predict', json=payload, timeout=10)
        result   = response.json()

        prediction  = result['prediction']
        probability = result['probability']

        if prediction == 'Yes':
            st.error(f'Prediction: Disease Detected  —  Confidence: {probability*100:.1f}%')
        else:
            st.success(f'Prediction: No Disease  —  Confidence: {(1-probability)*100:.1f}%')

    except Exception as e:
        st.error(f'Could not reach the API. Is the server running? ({e})')
