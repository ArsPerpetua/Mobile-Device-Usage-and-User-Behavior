import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from config import PARAM_GRID, RANDOM_STATE

# Adding project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def train_model(X_train, y_train):
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        PARAM_GRID,
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
    )
    grid_search.fit(X_train, y_train)
    print("\nHyperparameter terbaik:", grid_search.best_params_)
    return grid_search.best_estimator_
