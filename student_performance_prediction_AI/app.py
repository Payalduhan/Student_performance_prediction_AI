import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load models
pass_fail_model = pickle.load(open('pass_fail_model.pkl', 'rb'))
score_model = pickle.load(open('score_model.pkl', 'rb'))


st.title('🎓 Student Performance Prediction System')
st.write('Predict PASS/FAIL and Final Exam Score using AI')


study_hours = st.slider('Study Hours per Day', 1, 15)
attendance = st.slider('Attendance (%)', 50, 100)
previous_score = st.slider('Previous Exam Score', 30, 100)

if st.button('Predict Result'):

    input_data = pd.DataFrame(
        [[study_hours, attendance, previous_score]],
        columns=['study_hours', 'attendance', 'previous_score']
    )

    pass_fail_prediction = pass_fail_model.predict(input_data)[0]
    predicted_score = score_model.predict(input_data)[0]

    if pass_fail_prediction == 'Pass':
        st.success('Result: PASS ✅')
        st.info(f'Predicted Final Score: {predicted_score:.2f} / 100')
    else:
        st.error('Result: FAIL ❌')
        st.warning(f'Predicted Final Score: {predicted_score:.2f} / 100')


# Load dataset
data = pd.read_csv('student_data.csv')


st.subheader('📈 Data Insights')


fig, ax = plt.subplots()
ax.scatter(data['study_hours'], data['final_score'])
ax.set_xlabel('Study Hours')
ax.set_ylabel('Final Score')
st.pyplot(fig)
