import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample, shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from tensorflow import keras

# Load the data
data = pd.read_csv(r'C:\Users\najeh\OneDrive\Desktop\Project\Lending Club Loan Data Analysis\loan_data.csv')

# Check the first few rows of the dataset
print(data.head())

# Check the shape of the dataset
print(data.shape)

# Explore the dataset
print(data.describe())

# Check for missing values
print(data.isnull().any())

# Check the distribution of the target variable
print(data['not.fully.paid'].value_counts())

# Separate the classes
not_fully_paid_0 = data[data['not.fully.paid'] == 0]
not_fully_paid_1 = data[data['not.fully.paid'] == 1]

# Upsample the minority class
data_upsampled = resample(not_fully_paid_1, replace=True, n_samples=8045)
new_data = pd.concat([not_fully_paid_0, data_upsampled])

# Shuffle the dataset
new_data = shuffle(new_data)

# Encode categorical features
le = LabelEncoder()
for column in new_data.columns:
    if new_data[column].dtype == 'object':
        new_data[column] = le.fit_transform(new_data[column])

# Explore the updated dataset
print(new_data.head())

# Calculate correlation and visualize it
correlation_matrix = new_data.corr().abs()
print(correlation_matrix['not.fully.paid'].sort_values(ascending=False))

sns.heatmap(correlation_matrix, annot=True)

# Feature selection using SelectPercentile
sf = SelectPercentile(score_func=chi2, percentile=40)
x = new_data.iloc[:, :-1]
y = new_data.iloc[:, -1]
x = sf.fit_transform(x, y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Standardize the features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Build a neural network model with additional hidden layers
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(x_train.shape[1],)),  # Input layer
    keras.layers.Dense(512, activation='relu'), # Hidden layer 1 with 512 neurons and ReLU activation

    keras.layers.Dense(256, activation='relu'), # Hidden layer 2 with 256 neurons and ReLU activation
 
    keras.layers.Dense(128, activation='relu'), # Hidden layer 3 with 128 neurons and ReLU activation

    keras.layers.Dense(64, activation='relu'),  # Hidden layer 4 with 64 neurons and ReLU activation

    keras.layers.Dense(32, activation='relu'),  # Hidden layer 5 with 32 neurons and ReLU activation
  
    keras.layers.Dense(16, activation='relu'),  # Hidden layer 6 with 16 neurons and ReLU activation
    
    keras.layers.Dense(8, activation='relu'),   # Hidden layer 7 with 8 neurons and ReLU activation
    
    keras.layers.Dense(4, activation='relu'),   # Hidden layer 8 with 4 neurons and ReLU activation
    keras.layers.Dense(1, activation='sigmoid') # Output layer with sigmoid activation for binary classification
])
# Model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=200)


# Make predictions on the test set
predictions = (model.predict(x_test) > 0.5).astype('int32')

# Print classification report
print(classification_report(y_test, predictions))

# Print accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Test Accuracy: {accuracy * 100:.3f}%')