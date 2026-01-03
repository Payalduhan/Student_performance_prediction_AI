import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix


# Load data
data = pd.read_csv('student_data.csv')


X = data[['study_hours', 'attendance', 'previous_score']]
y = data['pass_fail']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Pass/Fail Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))


pickle.dump(model, open('pass_fail_model.pkl', 'wb'))


print('Pass/Fail model trained successfully')