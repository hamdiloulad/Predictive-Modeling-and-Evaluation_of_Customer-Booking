import pandas as pd
from oversampling import perform_oversampling
from scalling import scale_data
from splitData import split_data
from Encoder import encode_categorical
from randomForest import random_forest_predict

# Read the data from the CSV file
data = pd.read_csv('D:/DataProject/venv/PredictiveModel/import/customer_booking.csv', encoding='iso-8859-1')

# Encode categorical columns in the data
df_encoded = encode_categorical(data, columns=['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin'])
# Scale the data
scaled_data = scale_data(df_encoded)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(data=scaled_data, y="booking_complete", test_size=0.2, random_state=42)

# Perform oversampling on the training data
X_train_res, y_train_res = perform_oversampling(x=X_train, y=y_train) 

# Apply random forest for prediction and obtain the evaluation report
report = random_forest_predict(X_train_res, X_test, y_train_res, y_test, n_features_to_select=6)

# Print the evaluation report
print(report)
