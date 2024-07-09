

OUTLIERSSS

#| eval: false

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



df = pd.read_csv('weatherNewToPython.csv')

# Separate target variable
X = df.drop(['RainTomorrow'], axis=1)
#y = df['RainTomorrow'].copy()
y = df['RainTomorrow']
y.shape

#Training and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

#scaler = StandardScaler()
#scaled_X_train = scaler.fit_transform(X_train)
#scaled_X_test = scaler.transform(X_test)    

#print(X_train.shape, y_train.shape)

# Remove outliers
def cap_max_values(df, var, max_value):
  return np.where(df[var]>max_value, max_value, df[var])

##removing outliers from both training and testing set
for df in [X_train, X_test]:
  df['Rainfall'] = cap_max_values(df, "Rainfall", 3.2)
df['Evaporation'] = cap_max_values(df, "Evaporation", 21.8)
df['WindSpeed9am'] = cap_max_values(df, "WindSpeed9am", 55.0)
df['WindSpeed3pm'] = cap_max_values(df, "WindSpeed3pm", 57.0)

X_train[['Rainfall', 'Evaporation', 'WindSpeed9am', 'WindSpeed3pm']].describe().T
print(df.info())





MISSING VALUES


#| eval: false
numerics = ['int64', 'float64']
cat_vars = ['object']

def print_multiple_value(values: list):
  for index, value in enumerate(values):
  print(f"================================================{index + 1}================================================")
print("\n".join([
  str(value),
  ""
]))


# Numerical & Categorical columns 
#num_cols = list(X_train.select_dtypes(include=numerics).columns)
#cat_cols = list(X_train.select_dtypes(include=cat_vars).columns)

s = (df.dtypes == cat_vars)
cat_cols = list(s[s].index)


t = (df.dtypes == numerics)
num_cols = list(t[t].index)



# Imputing NUMERICAL
for df in [X_train, X_test]:
  for col in num_cols:
  col_median=X_train[col].median() # usign median to impute
df[col].fillna(col_median, inplace=True)

# Imputing CATEGORICAL
for df in [X_train, X_test]:
  for col in cat_cols:
  col_mode=X_train[col].mode()[0]
df[col].fillna(col_mode, inplace=True) 


# Check missing data
print_multiple_value([X_train.isnull().sum().T,X_test.isnull().sum().T])









ARTIFICIAL NEURAL NETWORKK!!!
  
  

  label_encoder = LabelEncoder()
for col in object_columns:
  df[col] = label_encoder.fit_transform(df[col]).astype(np.int64)
df.info()

X = df.drop(['RainTomorrow'],axis = 1)
y = df['RainTomorrow']
print(y.unique())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X.shape

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
print(len(y_train))
print(len(scaled_X_train))
print(len(scaled_X_test))
print(len(y_test))

# early stopping of model
early_stopping = callbacks.EarlyStopping(
  min_delta=0.001, 
  patience=10, 
  restore_best_weights=True,
)


# Initializing the Neural Network here
model = Sequential()

# Adjust the input_dim to match the number of features in your data
input_dim = scaled_X_train.shape[1]

# Adding layers to the network
model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim =input_dim))
model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
opt = Adam(learning_rate=0.001)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN
history = model.fit(X_train,y_train, batch_size = 32, epochs = 150, callbacks=[early_stopping], validation_split=0.2)


# Plotting Training and Validation accuracy
history_df = pd.DataFrame(history.history)

plt.figure(figsize =(10,4),dpi = 200)
plt.plot(history_df.loc[:, ['loss']], "Red", label='Training loss')
plt.plot(history_df.loc[:, ['val_loss']],"Green", label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")

plt.show()

print(len(y_train))
print(len(scaled_X_train))
print(len(scaled_X_test))
print(len(y_test))


# Predicting the test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

print(classification_report(y_test, y_pred))

