import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from qsar_hepatotoxicity_predictor.data import load_dataset, split_data
from qsar_hepatotoxicity_predictor.features import get_morgan_fingerprint

def build_meta_learner() -> VotingClassifier:
    """Builds a Soft Voting Meta-Learner using different Classification models."""
    base_models = [
        "rf", RandomForestClassifier(random_state=42),
        "knn", KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
        "svm", CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42), ensemble=False),
        "xgb", XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42)
    ]

    return VotingClassifier(estimators=base_models, voting='soft')

def training_pipeline(data_path: str, model_save_path: str):
    """Trains the Tox Khan dataset using a Soft Voting Meta-Learner."""
    # Load Data
    df = load_dataset(data_path)
    X_smiles, y = split_data(df)

    # Feature generation (Morgan Fingerprints)
    X_fp = get_morgan_fingerprint(X_smiles)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_fp, y, stratify=y, test_size=0.2, random_state=42)

    # Model Training
    model = build_meta_learner()
    model.fit(X_train, y_train)

    # Save the Trained Model
    joblib.dump(model, model_save_path)
    print(f"Model successfully trained and saved to {model_save_path}")

    return model