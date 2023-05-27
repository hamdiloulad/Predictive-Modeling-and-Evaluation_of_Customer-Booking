from sklearn.model_selection import train_test_split
def split_data(data,y, test_size, random_state):
    
    X = data.drop(columns=[y])  # Drop the target variable from the input features
    y = data[y]  # Set the target variable as the output
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test