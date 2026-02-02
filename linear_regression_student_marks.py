# Beginner ML Project:
# Linear Regression – Student Marks Prediction

# 1. Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 2. Create a simple dataset
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks":     [35, 40, 50, 60, 65, 70, 78, 85, 88, 92]
}

df = pd.DataFrame(data)

# 3. Explore the dataset
print("Dataset Preview:")
print(df.head(), "\n")

print("Dataset Info:")
print(df.info(), "\n")

print("Statistical Summary:")
print(df.describe(), "\n")

# 4. Define features (X) and label (y)
X = df[["StudyHours"]]   # Feature (input)
y = df["Marks"]          # Label (output)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2, "\n")

# 9. Predict marks for new data
new_hours = [[7.5]]
predicted_marks = model.predict(new_hours)

print(f"Predicted marks for studying {new_hours[0][0]} hours:", predicted_marks[0])
