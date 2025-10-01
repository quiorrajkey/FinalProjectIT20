import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC

st.set_page_config(page_title="Comparative Analysis of Machine Learning Algorithms for Cardiac Conditions Diagnosis", layout="wide")

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("data/Medicaldataset.csv")
    return df

def impute_and_clip(df_in):
    df = df_in.copy()
    if 'Gender' in df.columns:
        if pd.api.types.is_numeric_dtype(df['Gender']):
            df['Gender'] = df['Gender'].map({1: 'Male', 0: 'Female'}).fillna(df['Gender'])
        df['Gender'] = df['Gender'].fillna(df['Gender'].mode().iloc[0])

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df

def prepare_features(df):
    X = df.drop(columns=['Result'])
    y = df['Result']
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded, y

def train_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=42),
    }

    results = {}
    for name, model in models.items():
        if name in ["Decision Tree", "Random Forest", "Naive Bayes"]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=False)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            "model": model,
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm
        }
    return results, scaler

# ---------------------------
# UI Layout
# ---------------------------
st.markdown("<h1 style='display:flex;align-items:center;'>Comparative Analysis of Machine Learning Algorithms for Cardiac Conditions Diagnosis</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Dataset & Actions")
uploaded_file = st.sidebar.file_uploader("Upload Medicaldataset.csv (optional)", type=["csv"])
df = None
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        st.sidebar.success("Loaded uploaded CSV")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded file: {e}")

if df is None:
    try:
        df = load_data(None)
        st.sidebar.write("Loaded data/Medicaldataset.csv")
    except Exception as e:
        st.sidebar.error("No dataset found in data/Medicaldataset.csv and no file uploaded.")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.write("Quick dataset info:")
st.sidebar.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
if 'Result' not in df.columns:
    st.sidebar.error("Target column 'Result' not found. Please ensure your CSV has a 'Result' column.")
    st.stop()

if st.sidebar.checkbox("Show missing values (before imputation)"):
    st.sidebar.write(df.isnull().sum())

# ---------------------------
# Patient Input
# ---------------------------
st.subheader("Patient Input")
cols = st.columns([1,1,1])
with cols[0]:
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=50.0, step=1.0)
with cols[1]:
    gender = st.selectbox("Gender", options=["Male","Female"])
with cols[2]:
    heart_rate = st.number_input("Heart rate (bpm)", min_value=0.0, value=80.0, step=1.0)

cols = st.columns([1,1,1])
with cols[0]:
    sys_bp = st.number_input("Systolic blood pressure", min_value=0.0, value=130.0, step=1.0)
with cols[1]:
    dia_bp = st.number_input("Diastolic blood pressure", min_value=0.0, value=80.0, step=1.0)
with cols[2]:
    blood_sugar = st.number_input("Blood sugar (mg/dL)", min_value=0.0, value=100.0, step=1.0)

cols = st.columns([1,1,1])
with cols[0]:
    ck_mb = st.number_input("CK-MB", min_value=0.0, value=0.0, step=0.1)
with cols[1]:
    troponin = st.number_input("Troponin", min_value=0.0, value=0.0, step=0.01)
with cols[2]:
    extra_val = st.number_input("Extra value (optional)", min_value=0.0, value=0.0, step=0.1)

col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    preprocess_click = st.button("Preprocess & Train Models")
with col_b:
    show_hist = st.button("Show Histograms")
with col_c:
    predict_click = st.button("Predict for Input (after training)")

# ---------------------------
# Dataset Preview
# ---------------------------
with st.expander("Preview dataset (first 5 rows)"):
    st.dataframe(df.head())

if show_hist:
    st.subheader("Feature Histograms")
    selected = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure',
                'Blood sugar', 'CK-MB', 'Troponin']
    present = [col for col in selected if col in df.columns]
    n = len(present)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4*n))
    if n == 1: axes = [axes]
    for ax, feat in zip(axes, present):
        if df[feat].dtype == 'object':
            sns.countplot(x=feat, data=df, ax=ax)
        else:
            sns.histplot(df[feat].dropna(), kde=True, ax=ax, bins=30)
        ax.set_title(f"{feat}")
    st.pyplot(fig)

# ---------------------------
# Preprocess & Train
# ---------------------------
if preprocess_click:
    st.info("Preprocessing dataset and training models...")
    df_clean = impute_and_clip(df)
    X_encoded, y = prepare_features(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    results, scaler = train_models(X_train, X_test, y_train, y_test)

    st.success("Training complete. Models & metrics below.")
    for name, info in results.items():
        st.markdown(f"### {name} — Accuracy: {info['accuracy']:.4f}")
        with st.expander(f"Classification Report — {name}", expanded=False):
            st.text(info['report'])
        cm = info['confusion_matrix']
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.session_state['trained'] = True
    st.session_state['scaler'] = scaler
    st.session_state['X_encoded_columns'] = X_encoded.columns.tolist()
    st.session_state['X_full_scaled'] = scaler.transform(X_encoded)
    st.session_state['X_encoded'] = X_encoded
    st.session_state['df_clean'] = df_clean
    st.session_state['best_models'] = results

# ---------------------------
# Prediction
# ---------------------------
if predict_click:
    if not st.session_state.get('trained', False):
        st.error("Train models first before predicting.")
    else:
        input_dict = {
            'Age': age,
            'Gender': gender,
            'Heart rate': heart_rate,
            'Systolic blood pressure': sys_bp,
            'Diastolic blood pressure': dia_bp,
            'Blood sugar': blood_sugar,
            'CK-MB': ck_mb,
            'Troponin': troponin
        }
        input_df = pd.DataFrame([input_dict])
        X_cols = st.session_state['X_encoded_columns']
        input_encoded = pd.get_dummies(input_df, drop_first=True).reindex(columns=X_cols, fill_value=0)

        scaler = st.session_state['scaler']
        input_scaled = scaler.transform(input_encoded)

        st.write("Input features (encoded):")
        st.dataframe(input_encoded)

        preds = {}
        for name, info in st.session_state['best_models'].items():
            model = info['model']
            if name in ["Decision Tree", "Random Forest", "Naive Bayes"]:
                pred = model.predict(input_encoded)
                prob = None
            else:
                pred = model.predict(input_scaled)
                prob = model.predict_proba(input_scaled).max() if hasattr(model, "predict_proba") else None
            preds[name] = (pred[0], prob)

        st.subheader("Model predictions for this input")
        for name, (pred, prob) in preds.items():
            if prob is not None:
                st.write(f"**{name}** → Prediction: {pred} (prob={prob:.3f})")
            else:
                st.write(f"**{name}** → Prediction: {pred}")

        nbrs = NearestNeighbors(n_neighbors=4, metric='euclidean').fit(st.session_state['X_full_scaled'])
        distances, indices = nbrs.kneighbors(input_scaled)
        similar_rows = st.session_state['df_clean'].iloc[indices[0]]
        st.subheader("Top 3 similar patients")
        st.dataframe(similar_rows)

# ---------------------------
# Insights Section
# ---------------------------
st.markdown("## 🔎 Insights & Analysis")

if st.session_state.get('trained', False):
    results = st.session_state['best_models']
    accs = {name: info['accuracy'] for name, info in results.items()}
    best_model = max(accs, key=accs.get)
    worst_model = min(accs, key=accs.get)

    st.write(f"✅ Best performing model: **{best_model}** ({accs[best_model]:.2%} accuracy)")
    st.write(f"⚠️ Least performing model: **{worst_model}** ({accs[worst_model]:.2%} accuracy)")

    st.subheader("Model Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=list(accs.keys()), y=list(accs.values()), ax=ax, palette="viridis")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.bar_label(ax.containers[0], fmt="%.2f")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.subheader("Misclassification Insights")
    for name, info in results.items():
        cm = info['confusion_matrix']
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            st.write(f"- **{name}** → False Negatives: {fn}, False Positives: {fp}")
        else:
            st.write(f"- **{name}** → Multiclass confusion matrix, see above.")

    if 'scaler' in st.session_state and predict_click:
        st.subheader("Prediction Insights for Entered Patient")
        votes = []
        probs = []
        for name, info in results.items():
            model = info['model']
            if name in ["Decision Tree", "Random Forest", "Naive Bayes"]:
                pred = model.predict(input_encoded)
                prob = None
            else:
                pred = model.predict(input_scaled)
                prob = model.predict_proba(input_scaled).max() if hasattr(model,"predict_proba") else None
            votes.append(pred[0])
            if prob is not None: probs.append(prob)

        majority_vote = max(set(votes), key=votes.count)
        st.write(f"🩺 Majority of models predict: **{majority_vote}**")
        if probs:
            st.write(f"Average confidence across models: **{np.mean(probs):.2f}**")
else:
    st.info("Train models first to view insights.")

# ---------------------------
# End of app
# ---------------------------
st.markdown("---")
st.write("Notes: This app preprocesses and trains multiple classifiers. Use the dataset in `data/Medicaldataset.csv` or upload your own CSV. The target column name must be 'Result'.")
