# Basic Libraries
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ==================IMPORT DATASET=========================
original_df = pd.read_csv("misc\CTG_rawdata.csv")

# Check total rows vs unique SegFile entries
total_rows = original_df.shape[0]
unique_segfiles = original_df["SegFile"].nunique()

print(f"Total rows: {total_rows}")
print(f"Unique SegFile values: {unique_segfiles}")

# Check for duplicates
duplicates = original_df[original_df.duplicated(subset=["SegFile"], keep=False)]

cleaned_df = original_df.dropna().reset_index(drop=True)

keep_cols = [
    "b", "e", "LB", "LBE", "AC", "FM", "UC", "DS", "DP", "DR",
    "ASTV", "MSTV", "ALTV", "MLTV",
    "Width", "Min", "Max", "Nmax", "Nzeros",
    "Mode", "Mean", "Median", "Variance", "Tendency",
    "NSP"   # keep target
]

dropped_df = cleaned_df[keep_cols]

# Dictionary for renaming
histogram_rename = {
    "Width": "H_Width",
    "Min": "H_Min",
    "Max": "H_Max",
    "Nmax": "H_Nmax",
    "Nzeros": "H_Nzeros",
    "Mode": "H_Mode",
    "Mean": "H_Mean",
    "Median": "H_Median",
    "Variance": "H_Variance",
    "Tendency": "H_Tendency"
}

# Apply renaming to final_df
dropped_df = dropped_df.rename(columns=histogram_rename)

# Preview
dropped_df.head()

# ================CREATING NEW FEATURES==========================

final_df = dropped_df.copy()

# Create runtime column
final_df["runtime"] = original_df["e"] - original_df["b"]

# Event count features to normalize
event_features = ["AC", "FM", "UC", "DS", "DP", "DR"]

# Create normalized versions (per unit runtime)
for var in event_features:
    final_df[f"{var}_rate"] = final_df[var] / final_df["runtime"]

# Drop the old raw count columns
final_df = final_df.drop(columns=event_features)

final_df = final_df.drop(columns=["b", "e", "runtime"])

overall_df = final_df.copy()
normal_df = final_df[final_df["NSP"] == 1].reset_index(drop=True)
suspect_df = final_df[final_df["NSP"] == 2].reset_index(drop=True)
pathologic_df = final_df[final_df["NSP"] == 3].reset_index(drop=True)



# ================== DEFINE FEATURES AND TARGET ==================
X = overall_df.drop(columns=["NSP"])   # all predictors
y = overall_df["NSP"]                  # target labels

# ================== TRAIN/TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

from sklearn.preprocessing import LabelEncoder

# Encode labels for XGBoost
le = LabelEncoder()
y_train_sm_enc = le.fit_transform(y_train_sm)   # for training


# ================== Base Models ==================
rf_best = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=42
)

xgb_best = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

mlp_best = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        alpha=0.01,
        learning_rate_init=0.01,
        max_iter=500,
        activation="relu",
        solver="adam",
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    ))
])

# ================== Ensemble (Stacking) ==================
stack_model = StackingClassifier(
    estimators=[
        ('rf', rf_best),
        ('xgb', xgb_best),
        ('mlp', mlp_best)
    ],
    final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
    cv=5,
    n_jobs=-1
)

# ================== Train on SMOTE ==================
stack_model.fit(X_train_sm, y_train_sm)
