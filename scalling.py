from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def scale_data(df):
    
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Initialize the MinMaxScaler
    
    df_scaled = scaler.fit_transform(df)  # Scale the data using Min-Max scaling
    
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)  # Create a new DataFrame with scaled values
    
    return df_scaled