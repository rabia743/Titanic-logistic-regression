import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

titanic = pd.read_csv("titanic.csv")
# X = Independent Variables
X = titanic[["Passenger_Class", "Age", "Fare"]]
# y = Dependent Variable / Target
y = titanic["Survived"]
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# Create Model
model = LogisticRegression()
# Train Model
model.fit(X_train, y_train)
# predict on Test Data
test_prediction = model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, test_prediction)
print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")
# New Passenger
print("\nEnter New Passenger Details:")

Passenger_Class = int(input("Enter Passenger Class (1, 2, 3): "))
while True:
    Age = int(input("Enter Age: "))
    if Age > 0:
        break
    print("Age cannot be 0 or negative. Please enter a valid age.")
Fare = float(input("Enter Fare: "))
# Create New Passenger DataFrame
new_passenger = pd.DataFrame(
    [[Passenger_Class, Age, Fare]],
    columns=["Passenger_Class", "Age", "Fare"]
)
result = model.predict(new_passenger)
# probability
probability = model.predict_proba(new_passenger)

survived_probability = probability[0][1] * 100
not_survived_probability = probability[0][0] * 100
# Display Probability
print("\nSurvival Probability:", survived_probability, "%")
print("Not Survival Probability:", not_survived_probability, "%")

if result[0] == 1:
    print("Result: Survived")
else:
    print("Result: Did Not Survive")