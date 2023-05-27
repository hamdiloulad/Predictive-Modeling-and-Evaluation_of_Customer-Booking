from imblearn.over_sampling import RandomOverSampler

def perform_oversampling(x, y):
    
    # Instantiate the RandomOverSampler algorithm
    ros = RandomOverSampler()

    # Fit and apply RandomOverSampler to the training data
    X_train_resampled, y_train_resampled = ros.fit_resample(x, y)

    return X_train_resampled, y_train_resampled