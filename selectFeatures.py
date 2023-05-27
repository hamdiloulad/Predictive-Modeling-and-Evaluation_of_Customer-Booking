from sklearn.feature_selection import RFECV

def select_features(X_train_res, y_train_res, n_features_to_select,cv,model):

# Perform recursive feature elimination with cross-validation
    selector = RFECV(model, step=1, cv=cv, n_jobs=-1, scoring='f1_macro')
    selector.fit(X_train_res, y_train_res)

    # Get the selected features
    selected_features = X_train_res.columns[selector.get_support()][:n_features_to_select].tolist()
    return selected_features