import pandas as pd

def prediction_making(best_model, X_test):

    # Let's use the best model to make the final predictions and submission file creation.
    preds = best_model.predict(X_test)
    preds.shape

    SubmissionFormat = pd.read_csv('../data/raw/SubmissionFormat.csv')
    submission = pd.DataFrame({'id': SubmissionFormat["id"],'status_group': preds})
    submission.to_csv('../data/processed/submission.csv', index=False)
    print("Predictions completed")
