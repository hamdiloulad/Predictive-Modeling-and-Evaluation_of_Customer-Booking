from sklearn.ensemble import BaggingClassifier

def baggingClassifier(model, x, y):
    clf_bag = BaggingClassifier(base_estimator=model)
    clf_bag.fit(x, y)
    return clf_bag