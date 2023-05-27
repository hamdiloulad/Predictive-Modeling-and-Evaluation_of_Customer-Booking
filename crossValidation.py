from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

def cross_validate(model, x, y, selected_features, cv):

    # Perform cross-validation
    cv_scores = cross_val_score(model, x[selected_features], y, cv=cv, scoring='f1_macro')  
    
    # Plot results
    plt.plot(range(1, cv+1), cv_scores, marker='o')
    plt.xlabel('Number of folds')
    plt.ylabel('Cross-validation score')

    # Add horizontal dashed line in the background
    plt.grid(axis='y', linestyle='--', linewidth=0.5)

    # Save the chart as a PNG file
    plt.savefig("D:/DataProject/venv/PredictiveModel/result/cross_validation.png")
    plt.show()





