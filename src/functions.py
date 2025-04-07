import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Data Handling
def load_dataset(path: str):
    """Load a dataset from a given path."""
    return pd.read_csv(path)

def split_features_target(df, target_col):
    """Split dataframe into features and target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

# 2. Feature Selection 
def select_features(X_train, y_train, k=50, method='kbest'):
    """Select top features using SelectKBest or RFE."""
    if method == 'kbest':
        selector = SelectKBest(score_func=f_regression, k=k)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()]
    return X_train[selected_features], selected_features

# 3. Model Training 
def train_model(X_train, y_train, model):
    """Train the given model."""
    model.fit(X_train, y_train)
    return model

# 4. Model Prediction
def predict(model, X_test):
    """Make predictions using the trained model."""
    return model.predict(X_test)

# 5. Model Evaluation
def compute_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, and R² score."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2
    

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model on both train and test data."""
    metrics = {
        'train': compute_metrics(y_train, model.predict(X_train)),
        'test': compute_metrics(y_test, model.predict(X_test))
    }
    return metrics

def bootstrap_evaluation(model, X, y, n_iterations=100, random_state=42):
    """
    Perform bootstrap evaluation and return raw metric values.
    """
    np.random.seed(random_state)
    metrics = {'RMSE': [], 'MAE': [], 'R2': []}

    for _ in range(n_iterations):
        idx = np.random.choice(len(y), size=len(y), replace=True)
        X_sample, y_sample = X.iloc[idx], y.iloc[idx]
        y_pred = model.predict(X_sample)

        rmse, mae, r2 = compute_metrics(y_sample, y_pred)

        metrics['RMSE'].append(rmse)
        metrics['MAE'].append(mae)
        metrics['R2'].append(r2)

    return metrics

# 6. Hyperparameter Tuning
def tune_model(model, X_train, y_train, param_grid, scoring='neg_root_mean_squared_error', cv_splits=5):
    """
    Perform hyperparameter tuning using GridSearchCV.
    Returns the fitted GridSearchCV object.
    """
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring=scoring,
                               cv=cv,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

# 7. Visualization
def plot_metrics_boxplot(metrics_dict_by_model):
    """
    Create side-by-side boxplots for RMSE, MAE, and R2.
    metrics_dict_by_model = {
        'ModelA': {'RMSE': [...], 'MAE': [...], 'R2': [...]},
        'ModelB': {'RMSE': [...], 'MAE': [...], 'R2': [...]},
        ...
    }
    """
    metrics = ['RMSE', 'MAE', 'R2']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(metrics):
        data = []
        for model_name, model_scores in metrics_dict_by_model.items():
            for value in model_scores[metric]:
                data.append({'Model': model_name, 'Value': value})
        df = pd.DataFrame(data)
        sns.boxplot(ax=axes[i], x='Model', y='Value', data=df)
        axes[i].set_title(f'{metric} Distribution Across Models')
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=15)
        axes[i].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# 8. Model Persistence 
def save_model(model, path: str):
    """Save trained model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str):
    """Load trained model."""
    return joblib.load(path)
