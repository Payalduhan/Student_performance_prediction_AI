# 🎓 Student Performance Prediction System

An end-to-end 'Machine Learning + Streamlit' project that predicts whether a student is likely to 'Pass or Fail' and also estimates the 'final exam score' based on academic factors.

This project demonstrates real-world academic analytics using **classification and regression models** combined into a single intelligent system.


## 🧠 Problem Statement
Educational institutions often struggle to identify students who are at risk of failing.  
This system helps in:
- Early identification of weak students
- Predicting expected exam scores
- Supporting data-driven academic decisions


## 🤖 Machine Learning Models Used
- **Logistic Regression** → Pass / Fail Classification  
- **Linear Regression** → Final Score Prediction  

Both models are trained using **scikit-learn** and saved using **pickle**.

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib

## 📂 Project Structure
student-performance-prediction/
│
├── app.py
├── train_pass_fail.py
├── train_score_model.py
├── student_data.csv
├── pass_fail_model.pkl
├── score_model.pkl
├── requirements.txt
└── README.md

 ▶ How to Run the Project

 1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Train the models
python train_pass_fail.py
python train_score_model.py
3️⃣ Run the Streamlit app
streamlit run app.py
