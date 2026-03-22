import pandas as pd
import numpy as np
import librosa
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import joblib
from PIL import Image

# Constants
CSV_PATH = "/Users/ashutoshsharma/Desktop/Pakinsons/parkinsons1.csv"
DL_VOICE_DIR = "/Users/ashutoshsharma/Desktop/Pakinsons/sample of voice "
HW_DIR = "/Users/ashutoshsharma/Desktop/Pakinsons/Handwritten Samples"

# --- ML Voice Model Training (IMPROVED) ---
def train_ml_voice():
    print("Training ML Voice model (UCI dataset)...")
    df = pd.read_csv(CSV_PATH)
    X = df.drop(['name', 'status'], axis=1).values
    y = df['status'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try multiple models and pick the best
    models_to_try = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = 0
    best_model = None
    best_name = ''
    
    for name, model in models_to_try.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        print(f"{name}: {mean_score:.4f} (+/- {scores.std():.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with CV accuracy: {best_score:.2%}")
    
    # Train best model on full dataset
    best_model.fit(X_scaled, y)
    
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'accuracy': best_score,
        'model_name': best_name
    }, 'ml_voice_model.pkl')
    
    print(f"ML Voice model saved to ml_voice_model.pkl")

# --- DL Voice Model Training (SKIPPED) ---
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.network(x)

def train_dl_voice():
    pass

# --- Handwriting Model Training (SKIPPED) ---
def train_hw():
    pass

if __name__ == "__main__":
    train_ml_voice()
    # train_dl_voice()
    # train_hw()
    print("\nML MODEL OPTIMIZED AND SAVED")
