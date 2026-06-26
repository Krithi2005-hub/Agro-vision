# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
df = pd.read_csv("yield_df.csv")  # make sure the file name matches your saved file

# Encode the crop type (Item)
le = LabelEncoder()
df['Item'] = le.fit_transform(df['Item'])

# Prepare input features and target
X = df[['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Item']]
y = df['hg/ha_yield']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"✅ Model trained successfully! MSE: {mse:.2f}")

# Save the model and label encoder
joblib.dump(model, 'yield_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("✅ Model saved as 'yield_model.pkl' and encoder saved as 'label_encoder.pkl'")