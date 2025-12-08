import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Tensorflow / Keras for Model 9
import tensorflow as tf
from tensorflow import keras

# -----------------------------------------------------------------------------
# 6. Machine Learning: Setup & Preprocessing
# -----------------------------------------------------------------------------
st.header("6. Machine Learning Analysis")
st.divider()

st.subheader("Setup and Feature Selection")
st.markdown("""
To predict CVD events, we set up a machine learning pipeline. 
We defined our feature set including the calculated **$\Delta$Pulse Pressure** and baseline covariates.
""")

# 1. Define Features and Target
feature_cols = [
    "DELTA_PP", "V1_AGE", "V1_SEX", "V1_BMI",
    "V1_SYSBP", "V1_DIABP", "V1_GLUCOSE",
    "V1_TOTCHOL", "V1_CIGPDAY"
]

st.code(f"""
feature_cols = {feature_cols}
X = analytic[feature_cols]
y = analytic["CVD"]
""", language="python")

# Prepare X and y (Live execution)
# Ensure 'analytic' from previous section is available. 
# If running standalone, we filter the mock/loaded data again:
if 'analytic' not in locals():
    st.error("Please run the 'Analytic Dataset' section first to generate the data.")
else:
    X = analytic[feature_cols].copy()
    y = analytic["CVD"].copy()

    # Fill basic NaNs for the demo if any remain (Pipeline usually handles this)
    X = X.fillna(X.mean()) 

    st.subheader("Train/Test Split & Class Imbalance")
    st.markdown("""
    We split the data into training (80%) and testing (20%) sets, stratified by the CVD outcome to maintain class proportions.
    We also address the class imbalance (approx 1.5% CVD cases in original data) using **SMOTE** (Synthetic Minority Over-sampling Technique) within our pipeline.
    """)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=3002
    )

    with st.expander("👆 Expand to view Split & Pipeline code"):
        st.code("""
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=3002
)

# Preprocessing Pipeline structure
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', Classifier())
])
        """)

# -----------------------------------------------------------------------------
# HELPER: Unified Evaluation Function for the App
# -----------------------------------------------------------------------------
def app_evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Trains a model (live), makes predictions, and displays metrics in Streamlit.
    """
    with st.spinner(f"Training {name}..."):
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        # Check if model supports probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = np.zeros(len(y_pred)) # Fallback if no proba
            
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if hasattr(model, "predict_proba") else 0.5
        
        st.success(f"**{name} Trained Successfully!**")
        
        # Display Metrics
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{acc:.2%}")
        col2.metric("ROC AUC", f"{auc:.3f}")
        
        # Confusion Matrix
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No CVD', 'CVD'], yticklabels=['No CVD', 'CVD'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
        
    return model

# -----------------------------------------------------------------------------
# SECTION: ML Models
# -----------------------------------------------------------------------------
st.header("Machine Learning Models")
st.markdown("""
We tested 9 different models to predict CVD events. 
For each model, we used a **Pipeline** that includes:
1. **Preprocessing:** Standard Scaling for numeric features.
2. **Resampling:** SMOTE to handle the class imbalance.
3. **Classifier:** The specific algorithm.
""")

# -----------------------------------------------------------------------------
# Model 1: Logistic Regression
# -----------------------------------------------------------------------------
st.subheader("Model 1: Logistic Regression (Baseline)")
with st.expander("See Pipeline & Parameters"):
    st.code("""
# Pipeline
logreg_pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(solver="saga", max_iter=5000))
])

# Parameter Grid
logreg_param_grid = {
    "model__penalty": ["elasticnet"],
    "model__l1_ratio": [0, 0.5, 1],
    "model__C": [0.01, 0.1, 1, 10]
}
    """, language='python')

if st.button("Run Logistic Regression"):
    model = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(solver="saga", max_iter=1000, random_state=42))
    ])
    app_evaluate_model("Logistic Regression", model, X_train, y_train, X_test, y_test)

st.divider()

# -----------------------------------------------------------------------------
# Model 2: Decision Tree
# -----------------------------------------------------------------------------
st.subheader("Model 2: Decision Tree")
with st.expander("See Pipeline & Parameters"):
    st.code("""
# Pipeline
dt_pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", DecisionTreeClassifier(random_state=42))
])

# Parameter Grid
dt_param_grid = {
    "model__max_depth": [3, 5, 7, 10],
    "model__min_samples_leaf": [5, 10, 20],
    "model__ccp_alpha": [0.0, 0.001, 0.01]
}
    """, language='python')

if st.button("Run Decision Tree"):
    # Using specific params to avoid overfitting in demo
    model = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42))
    ])
    app_evaluate_model("Decision Tree", model, X_train, y_train, X_test, y_test)

st.divider()

# -----------------------------------------------------------------------------
# Model 3: Random Forest
# -----------------------------------------------------------------------------
st.subheader("Model 3: Random Forest")
with st.expander("See Pipeline & Parameters"):
    st.code("""
# Pipeline
rf_pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(random_state=42))
])

# Parameter Grid
rf_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [10, 15, None],
    "model__max_features": ["sqrt", "log2"],
    "model__max_samples": [0.7, 0.9]
}
    """, language='python')

if st.button("Run Random Forest"):
    model = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    app_evaluate_model("Random Forest", model, X_train, y_train, X_test, y_test)

st.divider()

# -----------------------------------------------------------------------------
# Model 4: K-Nearest Neighbors (KNN)
# -----------------------------------------------------------------------------
st.subheader("Model 4: K-Nearest Neighbors (KNN)")
with st.expander("See Pipeline & Parameters"):
    st.code("""
# Pipeline
knn_pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", KNeighborsClassifier())
])

# Parameter Grid
knn_param_grid = {
    "model__n_neighbors": [9, 15, 25],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2] # 1=Manhattan, 2=Euclidean
}
    """, language='python')

if st.button("Run KNN"):
    model = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", KNeighborsClassifier(n_neighbors=15))
    ])
    app_evaluate_model("KNN", model, X_train, y_train, X_test, y_test)

st.divider()

# -----------------------------------------------------------------------------
# Model 5: Support Vector Machine (SVM)
# -----------------------------------------------------------------------------
st.subheader("Model 5: Linear SVM")
with st.expander("See Pipeline & Parameters"):
    st.code("""
# Pipeline
svm_pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", SVC(probability=True, random_state=42))
])

# Parameter Grid
svm_param_grid = {
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": ["scale", 0.01, 0.1],
    "model__kernel": ["rbf"]
}
    """, language='python')

if st.button("Run SVM"):
    model = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", SVC(probability=True, C=1.0, kernel='rbf', random_state=42))
    ])
    app_evaluate_model("SVM", model, X_train, y_train, X_test, y_test)

st.divider()

# -----------------------------------------------------------------------------
# Model 6: Gradient Boosting
# -----------------------------------------------------------------------------
st.subheader("Model 6: Gradient Boosting Classifier")
with st.expander("See Pipeline & Parameters"):
    st.code("""
# Pipeline
gb_pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", GradientBoostingClassifier(random_state=42))
])

# Parameter Grid
gb_param_grid = {
    "model__n_estimators": [200, 500],
    "model__learning_rate": [0.01, 0.05],
    "model__max_depth": [3, 4],
    "model__subsample": [0.8, 1.0]
}
    """, language='python')

if st.button("Run Gradient Boosting"):
    model = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42))
    ])
    app_evaluate_model("Gradient Boosting", model, X_train, y_train, X_test, y_test)

st.divider()

# -----------------------------------------------------------------------------
# Model 7: Neural Network (MLP)
# -----------------------------------------------------------------------------
st.subheader("Model 7: Neural Network (MLP Sklearn)")
with st.expander("See Pipeline & Parameters"):
    st.code("""
# Pipeline
mlp_pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", MLPClassifier(early_stopping=True, max_iter=1000, random_state=42))
])

# Parameter Grid
mlp_param_grid = {
    "model__hidden_layer_sizes": [(64, 32), (100,)],
    "model__alpha": [0.001, 0.01, 0.1],
    "model__learning_rate_init": [0.001, 0.005]
}
    """, language='python')

if st.button("Run MLP (Sklearn)"):
    model = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    ])
    app_evaluate_model("MLP Neural Network", model, X_train, y_train, X_test, y_test)

st.divider()

# -----------------------------------------------------------------------------
# Model 8: Dummy Classifier (Baseline)
# -----------------------------------------------------------------------------
st.subheader("Model 8: Dummy Classifier (Reference)")
st.markdown("A baseline model that always predicts the most frequent class. Useful to check if other models are actually learning.")

if st.button("Run Dummy Classifier"):
    model = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", DummyClassifier(strategy="most_frequent"))
    ])
    app_evaluate_model("Dummy Classifier", model, X_train, y_train, X_test, y_test)

st.divider()

# -----------------------------------------------------------------------------
# Model 9: Neural Network (TensorFlow/Keras)
# -----------------------------------------------------------------------------
st.subheader("Model 9: Neural Network (TensorFlow)")
st.markdown("A custom Feed-Forward Neural Network built with Keras/TensorFlow.")

with st.expander("See Keras Model Builder Code"):
    st.code("""
def build_tf_model(input_dim, n_classes=2, learning_rate=0.001, 
                   hidden_units1=64, hidden_units2=32, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(hidden_units1, activation="relu"),
        keras.layers.Dense(hidden_units2, activation="relu"),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model
    """, language='python')

if st.button("Run TensorFlow Model"):
    with st.spinner("Training TensorFlow Model..."):
        # Data Prep (TensorFlow doesn't use the ImbPipeline the same way easily)
        # We manually scale and SMOTE for this demo
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)
        
        # Build Model
        input_dim = X_train_res.shape[1]
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["AUC"])
        
        # Train
        history = model.fit(X_train_res, y_train_res, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
        
        # Predict
        y_proba = model.predict(X_test_sc).flatten()
        y_pred = (y_proba > 0.5).astype(int)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        st.success("**TensorFlow Model Trained Successfully!**")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{acc:.2%}")
        col2.metric("ROC AUC", f"{auc:.3f}")
        
        # Plot Loss Curve
        st.write("Training History (Loss):")
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Val Loss')
        ax.legend()
        st.pyplot(fig)
