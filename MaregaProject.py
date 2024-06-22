import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
file_path = 'C:/Users/lumam/Desktop/Machine/weatherAUS.csv'
weather_data = pd.read_csv(file_path)

# Display the first few rows and summary information of the dataset
print(weather_data.head())
print(weather_data.info())

# Define numerical and categorical features
num_features = weather_data.select_dtypes(include=['float64']).columns.tolist()
cat_features = weather_data.select_dtypes(include=['object']).columns.tolist()

# Separate features and target
X = weather_data.drop(columns=['RainTomorrow'])
y = weather_data['RainTomorrow']

# Convert target variable to binary
y = y.map({'No': 0, 'Yes': 1})

# Impute missing values
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[num_features] = num_imputer.fit_transform(X[num_features])
X[cat_features] = cat_imputer.fit_transform(X[cat_features])

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=cat_features, drop_first=True)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))