from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from selectFeatures import select_features
from bagging import baggingClassifier
from featureImportance import plot_feature_importances
from crossValidation import cross_validate
from plotRocCurve import plot_roc_curve
from confusionMatrix import plot_confusion_matrix



def random_forest_predict(X_train_res, X_test, y_train_res, y_test, n_features_to_select, n_estimators=50, max_depth=5, cv=5):
    
    # Initialize the random forest classifier with the specified hyperparameters and class weights
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=5)
    
    # Perform feature selection
    selected_features = select_features(X_train_res, y_train_res, n_features_to_select, cv, model=clf)
    
    # Apply bagging with the selected features
    clf_bag = baggingClassifier(model=clf, x=X_train_res[selected_features], y=y_train_res)
    
    # Plot feature importances
    plot_feature_importances(model=clf_bag, x=X_train_res, selected_features=selected_features)
    
    # Perform cross-validation
    cross_validate(model=clf_bag, x=X_train_res, y=y_train_res, selected_features=selected_features, cv=cv)
    
    # Make predictions on the test data
    y_pred = clf_bag.predict(X_test[selected_features])
     
    # Visualize the confusion matrix 
    plot_confusion_matrix(y_test, y_pred)

    # Call the plot_roc_curve function
    plot_roc_curve(model=clf_bag, X_test=X_test[selected_features], y_test=y_test)
    
    # Compute the evaluation metrics of the model on the test data
    report = classification_report(y_test, y_pred)
    return report