import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime

def load_and_prepare_data(data_file):
    """
    Load the dataset and prepare it for training.
    
    Args:
        data_file: Path to the CSV file containing the dataset
        
    Returns:
        X: Feature dataframe
        y: Target variable
    """
    print(f"Loading dataset from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Display dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Features available: {len(df.columns) - 1}")  # Excluding the target column
    
    # Remove non-feature columns
    feature_cols = [col for col in df.columns if col not in ['Target', 'Season', 'Team1ID', 'Team2ID']]
    
    # Handle missing values
    missing_percentage = df[feature_cols].isnull().mean() * 100
    print(f"Columns with missing values: {(missing_percentage > 0).sum()}")
    
    X = df[feature_cols]
    y = df['Target']
    
    return X, y

def train_and_evaluate_model(X, y, model_type="men's"):
    """
    Train a LightGBM model and evaluate its performance.
    
    Args:
        X: Feature dataframe
        y: Target variable
        model_type: String indicating model type for display purposes
        
    Returns:
        model: Trained LightGBM model
        feature_importances: DataFrame of feature importances
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining {model_type} model")
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Create preprocessing pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Apply preprocessing
    X_train_processed = numeric_pipeline.fit_transform(X_train)
    X_test_processed = numeric_pipeline.transform(X_test)
    
    # Initialize and train LightGBM model
    model = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )
    
    # Train the model with a simple fit call to avoid parameter compatibility issues
    print("Training model...")
    model.fit(X_train_processed, y_train)
    
    # Make predictions on the test set
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = model.predict(X_test_processed)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    log_loss_score = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nModel Evaluation ({model_type}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {log_loss_score:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Extract feature importances
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Display top 15 most important features
    print("\nTop 15 Important Features:")
    print(feature_importances.head(15))
    
    # Create output directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save feature importances plot
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importances['feature'].head(15), feature_importances['importance'].head(15))
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importances ({model_type})')
    plt.tight_layout()
    plt.savefig(f'models/feature_importance_{model_type.replace("\'", "")}.png')
    
    # Save preprocessing pipeline
    with open(f'models/preprocessing_pipeline_{model_type.replace("\'", "")}.pkl', 'wb') as f:
        pickle.dump(numeric_pipeline, f)
    
    return model, feature_importances, (accuracy, log_loss_score, roc_auc)

def save_model(model, model_type):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        model_type: String indicating model type for filename
    """
    # Create directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model
    model_filename = f'models/{model_type}_model.txt'
    model.booster_.save_model(model_filename)
    print(f"Model saved to {model_filename}")

def main():
    print("=" * 80)
    print(f"NCAA Basketball Tournament Prediction Model Training")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Train men's model
    mens_data_file = 'mens_training_data.csv'
    X_men, y_men = load_and_prepare_data(mens_data_file)
    mens_model, mens_importances, mens_metrics = train_and_evaluate_model(X_men, y_men, "men's")
    save_model(mens_model, "mens")
    
    # Train women's model
    womens_data_file = 'womens_training_data.csv'
    X_women, y_women = load_and_prepare_data(womens_data_file)
    womens_model, womens_importances, womens_metrics = train_and_evaluate_model(X_women, y_women, "women's")
    save_model(womens_model, "womens")
    
    # Compare model performance
    print("\n" + "=" * 40)
    print("Model Performance Comparison")
    print("=" * 40)
    print(f"{'Metric':<15} {'Men''s Model':<15} {'Women''s Model':<15}")
    print(f"{'-' * 15:<15} {'-' * 15:<15} {'-' * 15:<15}")
    print(f"{'Accuracy':<15} {mens_metrics[0]:<15.4f} {womens_metrics[0]:<15.4f}")
    print(f"{'Log Loss':<15} {mens_metrics[1]:<15.4f} {womens_metrics[1]:<15.4f}")
    print(f"{'ROC AUC':<15} {mens_metrics[2]:<15.4f} {womens_metrics[2]:<15.4f}")
    
    print("\nModel training complete!")

if __name__ == "__main__":
    main()
