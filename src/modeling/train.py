"""Training routines for the sentiment classification model."""

import joblib
from sklearn.linear_model import LogisticRegression
from src.dataset import split_train_test  # adjust as needed for your import style


def train_model(df, test_size=0.2, random_state=42):
    """
    Train the sentiment analysis model on the training dataset.
    """
    x_train, _, y_train, _ = split_train_test(
        df,
        test_size=test_size,
        random_state=random_state
    )

    model = LogisticRegression(random_state=random_state)
    model.fit(x_train, y_train)
    joblib.dump(model, "model.joblib")

    logger = __import__("logging").getLogger(__name__)
    logger.info("Model training complete")

    return model
