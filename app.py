
import streamlit as st
st.set_page_config(page_title="Project Model Builder — Polished", layout="wide", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc, classification_report)

# ---------- Helper functions ----------
def load_csv(uploaded_file):
    """Read uploaded file (CSV/XLSX) safely."""
    try:
        if uploaded_file is None:
            return None
        if uploaded_file.name.lower().endswith(".xlsx") or uploaded_file.name.lower().endswith(".xls"):
            return pd.read_excel(uploaded_file)
        else:
            # read csv; allow various encodings by trying default then utf-8-sig
            try:
                return pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

def get_problem_type(series):
    return "regression" if pd.api.types.is_numeric_dtype(series) else "classification"

def basic_preprocess(df, features, num_strategy="mean", cat_strategy="most_frequent", scale=True):
    X = df[features].copy()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        num_imp = SimpleImputer(strategy=num_strategy)
        steps = [("imputer", num_imp)]
        if scale:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric_cols))
    if categorical_cols:
        if cat_strategy == "most_frequent":
            cat_imp = SimpleImputer(strategy="most_frequent")
        else:
            cat_imp = SimpleImputer(strategy="constant", fill_value="missing")
        # Use sparse_output for newer sklearn, works for older if attribute ignored
        cat_pipe = Pipeline([("imputer", cat_imp), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        transformers.append(("cat", cat_pipe, categorical_cols))
    preproc = ColumnTransformer(transformers=transformers, remainder="drop")
    return preproc, numeric_cols, categorical_cols

def plot_corr_heatmap(df, numeric_cols):
    if not numeric_cols:
        st.info("No numeric columns for correlation heatmap.")
        return
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def to_excel_bytes(obj):
    buffer = BytesIO()
    obj.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer

def is_continuous_target(series, uniq_threshold=20):
    if pd.api.types.is_numeric_dtype(series):
        return series.nunique() > uniq_threshold
    return False

# ---------- UI ----------
st.title("🚀 PREDICTIVE MODELLING OF LIFE EXPECTANCY USING REGRESSION AND CLASSIFICATION ")
st.write("Upload your CSV, pick a target column, train multiple models, compare metrics, visualize results, and export the best model.")

with st.sidebar:
    st.header("Step 1 — Upload / Sample")
    uploaded_file = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])
    use_sample = st.checkbox("Or use built-in sample dataset (Iris / California)", value=False)
    st.markdown("---")
    st.header("Step 2 — Preprocess")
    num_strategy = st.selectbox("Numeric missing strategy", options=["mean", "median", "drop rows"], index=0)
    cat_strategy = st.selectbox("Categorical missing strategy", options=["most_frequent", 'constant ("missing")'], index=0)
    scale_numeric = st.checkbox("Scale numeric features", value=True)
    st.markdown("---")
    st.header("Step 3 — Training")
    rand_state = st.number_input("Random state", value=42, step=1)
    do_hyper = st.checkbox("Run quick randomized hyperparam search (slower)", value=False)
    st.markdown("---")
    st.header("Step 4 — Export")
    export_model = st.checkbox("Allow model export button after training", value=True)

# Load dataset
if uploaded_file is None and not use_sample:
    st.info("Upload your dataset or choose sample dataset from the sidebar to get started.")
    st.stop()

if use_sample:
    sample_choice = st.selectbox("Sample dataset", options=["Iris (classification)", "California Housing (regression)"])
    if sample_choice.startswith("Iris"):
        from sklearn.datasets import load_iris
        ir = load_iris(as_frame=True)
        df = ir.frame
        df["target"] = ir.target
        st.success("Loaded Iris sample dataset.")
    else:
        from sklearn.datasets import fetch_california_housing
        ch = fetch_california_housing(as_frame=True)
        df = ch.frame
        st.success("Loaded California Housing sample dataset.")
else:
    df = load_csv(uploaded_file)
    if df is None:
        st.stop()

# Show top info
st.subheader("Dataset preview & info")
col1, col2 = st.columns([2,1])
with col1:
    st.dataframe(df.head(8))
with col2:
    st.write("Shape:")
    st.write(df.shape)
    st.write("Columns:")
    st.write(list(df.columns))
    missing = df.isna().sum()
    top_missing = missing[missing>0].sort_values(ascending=False)
    if len(top_missing):
        st.write("Columns with missing values (top):")
        st.write(top_missing.head(6))
    else:
        st.write("No missing values detected.")

# Target & features selection
all_cols = df.columns.tolist()
default_target = all_cols[-1]
target_col = st.selectbox("Select target column", options=all_cols, index=all_cols.index(default_target))
features = st.multiselect("Select features (leave blank = use all except target)", options=[c for c in all_cols if c!=target_col])
if not features:
    features = [c for c in all_cols if c!=target_col]

# Problem type auto detect (and override)
auto_type = get_problem_type(df[target_col])
st.write(f"Auto-detected problem type for target **{target_col}**: **{auto_type}**")
problem_type = st.selectbox("Choose problem type (override if needed)", options=[auto_type, "regression" if auto_type=="classification" else "classification"], index=0)
problem_type = problem_type  # final chosen

# Display correlation heatmap toggle
if st.checkbox("Show correlation heatmap (numeric features)"):
    numeric_cols_for_corr = df[features].select_dtypes(include=['number']).columns.tolist()
    plot_corr_heatmap(df, numeric_cols_for_corr)

# Build preprocessing
preprocessor, numeric_cols, categorical_cols = basic_preprocess(df, features,
                                                               num_strategy = num_strategy if num_strategy!="drop rows" else "mean",
                                                               cat_strategy = "most_frequent" if cat_strategy.startswith("most_frequent") else "constant",
                                                               scale = scale_numeric)

# Prepare X, y (handle drop rows option)
if num_strategy == "drop rows" or cat_strategy.startswith("drop rows"):
    df_work = df[features + [target_col]].dropna()
else:
    df_work = df[features + [target_col]].copy()

X = df_work[features]
y = df_work[target_col]

# Validate target vs chosen problem type
if problem_type == "classification" and is_continuous_target(y):
    st.warning(
        "Detected continuous numeric target but you selected 'classification'. Auto-switching to 'regression'.\n"
        "If you truly want classification, bin the target into classes (see code comments in app)."
    )
    problem_type = "regression"

# Label encoding (only if classification and target is categorical)
label_enc = None
if problem_type == "classification":
    if y.dtype == 'O' or y.dtype.name == 'category':
        label_enc = LabelEncoder()
        y_encoded = label_enc.fit_transform(y)
    else:
        # numeric but considered categorical (few unique values) -> use as-is
        y_encoded = y
else:
    y_encoded = y

# Train/Test split
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=int(rand_state))

# Models to train UI
st.subheader("Model selection & training")
colA, colB = st.columns(2)
with colA:
    if problem_type == "regression":
        chosen_models = st.multiselect("Choose models to train", options=["LinearRegression","ElasticNet","Lasso","RandomForestRegressor"], default=["RandomForestRegressor","ElasticNet"])
    else:
        chosen_models = st.multiselect("Choose models to train", options=["LogisticRegression","RandomForestClassifier","GradientBoostingClassifier"], default=["RandomForestClassifier","LogisticRegression"])
with colB:
    train_button = st.button("Train selected models (70/30 split)")

# Function to get model object & param grid (always returns a tuple)
def get_model_and_grid(name, problem):
    if problem == "regression":
        if name == "LinearRegression":
            return LinearRegression(), {}
        if name == "ElasticNet":
            return ElasticNet(random_state=int(rand_state)), {"model__alpha": np.logspace(-4, 1, 20), "model__l1_ratio": [0.1,0.5,0.9]}
        if name == "Lasso":
            return Lasso(random_state=int(rand_state)), {"model__alpha": np.logspace(-4, 0, 20)}
        if name == "RandomForestRegressor":
            return RandomForestRegressor(random_state=int(rand_state)), {"model__n_estimators":[50,100,200], "model__max_depth":[None,5,10,20]}
        # fallback
        st.warning(f"Unknown regression model '{name}'. Falling back to RandomForestRegressor.")
        return RandomForestRegressor(random_state=int(rand_state)), {"model__n_estimators":[100], "model__max_depth":[None]}
    else:
        if name == "LogisticRegression":
            return LogisticRegression(max_iter=2000, random_state=int(rand_state)), {"model__C": np.logspace(-3,3,20)}
        if name == "RandomForestClassifier":
            return RandomForestClassifier(random_state=int(rand_state)), {"model__n_estimators":[50,100,200], "model__max_depth":[None,5,10,20]}
        if name == "GradientBoostingClassifier":
            return GradientBoostingClassifier(random_state=int(rand_state)), {"model__n_estimators":[50,100,200], "model__learning_rate":[0.01,0.05,0.1], "model__max_depth":[3,5,7]}
        # fallback
        st.warning(f"Unknown classification model '{name}'. Falling back to RandomForestClassifier.")
        return RandomForestClassifier(random_state=int(rand_state)), {"model__n_estimators":[100], "model__max_depth":[None]}

# ---------- Robust training loop ----------
results = []
trained_pipelines = {}
if train_button:
    if not chosen_models:
        st.warning("Select at least one model to train.")
    else:
        st.write("Chosen models:", chosen_models)
        progress = st.progress(0)
        total = len(chosen_models)
        for idx, name in enumerate(chosen_models):
            res = None
            try:
                res = get_model_and_grid(name, problem_type)
            except Exception as e:
                st.error(f"Error calling get_model_and_grid for '{name}': {e}")
                res = None

            if res is None:
                st.warning(f"get_model_and_grid returned None for model '{name}'. Skipping this model.")
                progress.progress(int((idx+1)/total * 100))
                continue
            if not (isinstance(res, (list, tuple)) and len(res) == 2):
                st.warning(f"get_model_and_grid returned unexpected value for '{name}': {res}. Skipping.")
                progress.progress(int((idx+1)/total * 100))
                continue

            model_obj, grid = res
            pipeline = Pipeline([("preproc", preprocessor), ("model", model_obj)])
            info = ""

            # hyperparameter search optional
            if do_hyper and grid:
                try:
                    n_iter = min(10, max(1, len(list(grid.keys())) * 3))
                    rs = RandomizedSearchCV(pipeline, param_distributions=grid, n_iter=n_iter, cv=3, random_state=int(rand_state), n_jobs=-1)
                    rs.fit(X_train, y_train)
                    pipeline = rs.best_estimator_
                    info = f"RandomSearch done ({name})"
                except Exception as e:
                    st.warning(f"Hyperparameter search failed for {name}: {e}. Falling back to default fit.")
                    pipeline.fit(X_train, y_train)
                    info = f"Fitted (no-search fallback) ({name})"
            else:
                pipeline.fit(X_train, y_train)
                info = f"Fitted ({name})"

            # predictions & metrics
            try:
                y_pred = pipeline.predict(X_test)
            except Exception as e:
                st.error(f"Prediction failed for {name}: {e}")
                progress.progress(int((idx+1)/total * 100))
                continue

            if problem_type == "regression":
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
            else:
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                roc = None
                if hasattr(pipeline.named_steps['model'], "predict_proba"):
                    try:
                        probs = pipeline.predict_proba(X_test)
                        if probs.shape[1] == 2:
                            roc = roc_auc_score(y_test, probs[:,1])
                    except Exception:
                        roc = None
                metrics = {"Accuracy": acc, "F1-weighted": f1}
                if roc is not None:
                    metrics["ROC-AUC"] = roc

            results.append({"model": name, "metrics": metrics, "pipeline": pipeline, "info": info})
            trained_pipelines[name] = pipeline
            progress.progress(int((idx+1)/total * 100))
        st.success("Training completed for selected models.")

# Show comparison table & details
if results:
    st.subheader("Model comparison")
    rows = []
    for r in results:
        m = r["metrics"]
        row = {"Model": r["model"]}
        row.update(m)
        rows.append(row)
    comp_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(comp_df.style.format("{:.4f}"))

    if problem_type == "regression":
        best_model_name = comp_df["RMSE"].idxmin()
    else:
        if "ROC-AUC" in comp_df.columns:
            best_model_name = comp_df["ROC-AUC"].idxmax()
        else:
            best_model_name = comp_df["Accuracy"].idxmax()
    st.info(f"Best model (by chosen metric): **{best_model_name}**")
    best_pipeline = None
    for r in results:
        if r["model"] == best_model_name:
            best_pipeline = r["pipeline"]
            break

    st.subheader(f"Detailed results — {best_model_name}")
    tab1, tab2, tab3 = st.tabs(["Metrics & Plots", "Feature importance", "Confusion / ROC"])
    with tab1:
        st.write("Metrics:")
        st.json([r for r in results if r["model"]==best_model_name][0]["metrics"])
        if problem_type == "regression":
            y_pred = best_pipeline.predict(X_test)
            fig = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual', 'y':'Predicted'}, title="Actual vs Predicted")
            fig.add_shape(x0=min(y_test), x1=max(y_test), y0=min(y_test), y1=max(y_test),
                          line=dict(dash="dash", color="red"))
            st.plotly_chart(fig, use_container_width=True)
            resid = y_test - y_pred
            fig2 = px.histogram(resid, nbins=30, title="Residuals distribution", labels={'value':'Residual'})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            y_pred = best_pipeline.predict(X_test)
            st.text("Classification report:")
            st.text(classification_report(y_test, y_pred))
    with tab2:
        model_obj = best_pipeline.named_steps['model']
        try:
            feature_names = []
            pre = best_pipeline.named_steps['preproc']
            if hasattr(pre, "transformers_"):
                for name, trans, cols in pre.transformers_:
                    if name == "num":
                        # cols is list of numeric column names
                        feature_names.extend(cols)
                    elif name == "cat":
                        # trans is a Pipeline with an OneHotEncoder step
                        try:
                            ohe = trans.named_steps['ohe']
                            names = ohe.get_feature_names_out(cols)
                            feature_names.extend(list(names))
                        except Exception:
                            feature_names.extend(cols)
            else:
                feature_names = features

            if hasattr(model_obj, "feature_importances_"):
                fi = model_obj.feature_importances_
                fi_series = pd.Series(fi, index=feature_names).sort_values(ascending=False).head(20)
                fig = px.bar(fi_series, x=fi_series.values, y=fi_series.index, orientation='h', title="Feature importances (top 20)")
                st.plotly_chart(fig, use_container_width=True)
            elif hasattr(model_obj, "coef_"):
                coefs = model_obj.coef_
                if np.ndim(coefs) > 1:
                    coefs = np.mean(np.abs(coefs), axis=0)
                coef_series = pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=False).head(20)
                fig = px.bar(coef_series, x=coef_series.values, y=coef_series.index, orientation='h', title="Model coefficients (top 20)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")
        except Exception as e:
            st.error(f"Could not compute feature importances: {e}")

    with tab3:
        if problem_type == "classification":
            y_pred = best_pipeline.predict(X_test)
            st.write("Confusion matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, title="Confusion matrix")
            st.plotly_chart(fig, use_container_width=True)
            if hasattr(best_pipeline.named_steps['model'], "predict_proba"):
                try:
                    probs = best_pipeline.predict_proba(X_test)
                    if probs.shape[1] == 2:
                        fpr, tpr, _ = roc_curve(y_test, probs[:,1])
                        auc_val = auc(fpr, tpr)
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc_val:.3f}"))
                        fig2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash='dash'), showlegend=False))
                        fig2.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR")
                        st.plotly_chart(fig2, use_container_width=True)
                except Exception:
                    st.info("ROC curve could not be computed (maybe multiclass or no prob).")
        else:
            st.info("Confusion / ROC not applicable for regression.")

    # Prediction form for best model
    st.subheader("Interactive prediction (use one row of inputs)")
    predict_cols = features
    with st.form("pred_form"):
        pred_input = {}
        for c in predict_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                val = st.number_input(c, value=float(np.nan_to_num(df[c].median())))
            else:
                uniques = df[c].dropna().unique().tolist()
                if len(uniques) <= 10:
                    val = st.selectbox(c, options=uniques, index=0)
                else:
                    val = st.text_input(c, value=str(uniques[0]) if uniques else "")
            pred_input[c] = val
        sub = st.form_submit_button("Predict with best model")
    if sub:
        in_df = pd.DataFrame([pred_input])
        try:
            pred = best_pipeline.predict(in_df)
            if problem_type=="classification" and label_enc is not None:
                try:
                    pred_label = label_enc.inverse_transform([int(pred[0])])[0]
                    st.success(f"Predicted class: {pred_label}")
                except Exception:
                    st.success(f"Predicted (encoded): {pred[0]}")
            else:
                st.success(f"Prediction: {pred[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Export model
    if export_model:
        buf = BytesIO()
        joblib.dump(best_pipeline, buf)
        buf.seek(0)
        st.download_button(label="Download trained model (.joblib)", data=buf, file_name=f"{best_model_name}_model.joblib", mime="application/octet-stream")

# If no results yet, friendly note
if not results:
    st.info("Train models using the 'Train selected models' button. App will show model comparison, plots and allow prediction & export.")

st.markdown("---")

