import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances(model, selected_features, x):
    # Calculate feature importances
    importances = np.mean([estimator.feature_importances_ for estimator in model.estimators_], axis=0)

    # Plot feature importances
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x[selected_features].shape[1]), importances, color="r", align="center")
    plt.xticks(range(x[selected_features].shape[1]), selected_features, rotation=90)
    plt.xlim([-1, x[selected_features].shape[1]])

    # Add horizontal dashed line in the background
    for y in np.arange(0, 1.1, 0.1):
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)

    plt.savefig('D:/DataProject/venv/PredictiveModel/result/feature_importance.png')
    plt.close()
