import streamlit as st
import pandas as pd
import joblib

model=joblib.load('KNN_heart.pkl')
scaler=joblib.load('scaler.pkl')
expected_columns=joblib.load('columns.pkl')

st.title('Heart Disease Prediction by Kashish')
st.markdown('## Input your data to predict heart disease')
age=st.slider('Age',18,100,40)
sex=st.selectbox('Sex',['Male','Female'])
chest_pain=st.selectbox('Chest Pain',['ATA','NAP','ASY','TA'])
resting_blood_pressure=st.number_input('Resting Blood Pressure',0,200,120)
cholesterol=st.number_input('Cholesterol',100,600,200)
fasting_blood_sugar=st.selectbox('Fasting Blood Sugar',['0','1'])
resting_ecg=st.selectbox('Resting ECG',['Normal','ST','LVH'])
max_heart_rate=st.number_input('Max Heart Rate',60,220,150)
exercise_induced_angina=st.selectbox('Exercise Induced Angina',['Y','N'])
oldpeak=st.number_input('Oldpeak',0.0,6.0,1.0)
st_slope=st.selectbox('ST Slope',['Up','Flat','Down'])
if st.button('Predict'):
    raw_input={
        'age':age,
        'sex':sex,
        'chest_pain':chest_pain,
        'resting_blood_pressure':resting_blood_pressure,
        'cholesterol':cholesterol,
        'fasting_blood_sugar':fasting_blood_sugar,
        'resting_ecg':resting_ecg,
        'max_heart_rate':max_heart_rate,
        'exercise_induced_angina':exercise_induced_angina,
    }
    input_df=pd.DataFrame([raw_input])

    for column in expected_columns:
        if column not in input_df.columns:
            input_df[column]=0
    input_df=input_df[expected_columns]
    scaled_input=scaler.transform(input_df)
    prediction=model.predict(scaled_input)[0]
    if prediction==1:
        st.error('You have a high risk of heart disease')
    else:
        st.success('You are healthy')



