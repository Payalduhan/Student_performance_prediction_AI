import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import mean_absolute_error, r2_score


# Load data
data = pd.read_csv('student_data.csv')


X = data[['study_hours', 'attendance', 'previous_score']]
y = data['final_score']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")
pickle.dump(model, open('score_model.pkl', 'wb'))


print('Score prediction model trained successfully')


