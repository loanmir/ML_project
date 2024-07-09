import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("C:\\Users\\Sasa\\Desktop\\Univerza\\Machine\\Project_implementation\\weatherAUS.csv")

# Define columns with missing values
numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Fill missing values
for col in numeric_features:
    data[col] = data[col].fillna(data[col].median())

for col in categorical_features:
    data[col] = data[col].fillna(data[col].mode()[0])

# Prepare features and labels
X = data.drop(columns=['RainTomorrow'])
y = data['RainTomorrow']

# Ensure that 'RainTomorrow' is not included in categorical_features
categorical_features = [col for col in categorical_features if col != 'RainTomorrow']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create CatBoost Pool
train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_features)

# Initialize CatBoost model
model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=True)

# Train the model
model.fit(train_pool)

# Get evaluation results
evals_result = model.get_evals_result()

# Print the loss at each iteration
losses = evals_result['learn']['Logloss']
for i, loss in enumerate(losses):
    print(f'Iteration {i}: Loss = {loss}')

# Make predictions
preds = model.predict(test_pool)

# Evaluate the model
accuracy = (preds == y_test).mean()
print(f'Accuracy: {accuracy}')

print(classification_report(y_test, preds))



