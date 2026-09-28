import pandas as pd

ship = pd.read_csv("titanic.csv")
print(ship.head(10))
print(ship.tail(10))
print(ship.describe())
print(ship.columns)
# print(ship.to_string())

# DATA CLEANING
# Detect the missing values
print(ship.isnull().sum())
# =========================
# HANDLE MISSING VALUES
# =========================

# Age ko numeric mein convert karo
ship["Age"] = pd.to_numeric(ship["Age"], errors="coerce")
# Age ke missing values ko mean se fill karo
ship["Age"] = ship["Age"].fillna(ship["Age"].mean())
# Embarked ke missing values ko mode se fill karo
ship["Embarked"] = ship["Embarked"].fillna(ship["Embarked"].mode()[0])
# Cabin ke missing values ko "Unknown" se fill karo
ship["Cabin"] = ship["Cabin"].fillna("Unknown")
# Maine Add Kiya:
ship["Parents_Children"] = ship["Parents_Children"].fillna(0).astype(int)
# sbilings_Spouse ke missing values ko 0 se fill karo
ship["Siblings_Spouse"] = ship["Siblings_Spouse"].fillna(0).astype(int)
# Check missing values again
print(ship.isnull().sum())

# =========================
# CHECK DUPLICATE VALUES
# =========================
print("Before:", ship.shape)
print("Duplicate values:", ship.duplicated().sum())
# Remove duplicate values
ship.drop_duplicates(inplace=True)
print("After:", ship.shape)
print("Duplicate values:", ship.duplicated().sum())


# CHECK ROWS AND COLUMNS
print(ship.shape)


# CHECK DATA TYPES
print(ship.dtypes)

# CHANGE DATA TYPE
ship["Age"] = ship["Age"].astype(int)
print(ship.dtypes)

# rename columns
ship.rename(columns={"Pclass": "Passenger_Class", "SibSp": "Siblings_Spouse", "Parch": "Parents_Children","Sex": "Gender"}, inplace=True)

# TEXT DATA CLEANING
ship["Gender"] = ship["Gender"].str.strip().str.lower()
ship["Embarked"] = ship["Embarked"].str.strip().str.upper()



# CHECK CLEANED DATA
print(ship.head())
print(ship.isnull().sum())
print(ship.dtypes)

# SAVE CLEANED DATA BACK TO CSV FILE
ship.to_csv("titanic.csv", index=False)
print("Cleaned data saved successfully!")