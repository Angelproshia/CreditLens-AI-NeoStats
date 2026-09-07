from pathlib import Path
import gdown
import pandas as pd
import plotly.express as px
import streamlit as st
from src.config import DATA_PATH, DATA_URL, MODEL_PATH
from src.data import load_data, quality_summary, select_features
from src.model import train_model, load_bundle, predict_one, explain_linear_prediction
from src.talk_to_data import ask, EXAMPLES

st.set_page_config(page_title="CreditLens AI", page_icon="◉", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem}.hero{background:linear-gradient(120deg,#071a2d,#0f766e);padding:28px;border-radius:18px;color:white}
.hero h1{margin:0;font-size:2.5rem}.small{opacity:.82}.stMetric{background:#152536;padding:16px;border-radius:12px;border:1px solid #29465f}
[data-testid="stMetricLabel"] p{color:#b9d7e8!important}[data-testid="stMetricValue"]{color:#fff!important}
</style><div class='hero'><h1>CreditLens AI</h1><div class='small'>Explainable credit-risk intelligence for faster, auditable decisions</div></div>""", unsafe_allow_html=True)

if not Path(DATA_PATH).exists():
    try:
        Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("Downloading the Home Credit dataset for first-time setup..."):
            downloaded = gdown.download(DATA_URL, str(DATA_PATH), quiet=True, fuzzy=True)
        if not downloaded or not Path(DATA_PATH).exists():
            raise RuntimeError("Google Drive did not return the dataset file.")
    except Exception as exc:
        st.error(f"Dataset setup failed: {exc}")
        st.info("Confirm that the Google Drive dataset permission is set to ‘Anyone with the link – Viewer’.")
        st.stop()

@st.cache_resource(show_spinner=False)
def get_data(): return load_data(DATA_PATH)

df = get_data()
tabs = st.tabs(["Overview & EDA", "Risk prediction", "Explainability & rules", "Talk to data", "Model performance"])

with tabs[0]:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Applications", f"{len(df):,}")
    c2.metric("Model features", f"{len(select_features(df))}")
    c3.metric("Default rate", f"{df.TARGET.mean()*100:.2f}%")
    c4.metric("Duplicate rows", f"{df.duplicated().sum():,}")
    left,right = st.columns(2)
    with left:
        chart_df = df.sample(min(50000, len(df)), random_state=42)
        st.plotly_chart(px.histogram(chart_df, x="AMT_CREDIT", color="TARGET", nbins=45, barmode="overlay", title="Credit amount distribution (50k-row visual sample)"), use_container_width=True)
    with right:
        grouped=df.groupby("NAME_INCOME_TYPE",dropna=False).TARGET.agg(["mean","size"]).reset_index().sort_values("size",ascending=False).head(10)
        grouped["default_rate_pct"]=grouped["mean"]*100
        st.plotly_chart(px.bar(grouped,x="NAME_INCOME_TYPE",y="default_rate_pct",hover_data=["size"],title="Default rate by income type"),use_container_width=True)
    st.subheader("Five business insights to validate")
    st.markdown("1. Compare external credit scores with default outcome.  2. Identify high-risk income types.  3. Review credit-to-income burden.  4. Compare age and employment stability.  5. Track missingness as a possible process-quality signal.")
    with st.expander("Data-quality report"):
        st.dataframe(quality_summary(df).head(40), use_container_width=True)

if not Path(MODEL_PATH).exists():
    with st.spinner("Training the first model (one-time setup)..."):
        bundle=train_model(df,select_features(df),MODEL_PATH)
else: bundle=load_bundle(MODEL_PATH)

with tabs[1]:
    st.caption("Select an existing applicant to demonstrate end-to-end scoring; values remain editable below.")
    idx=st.number_input("Dataset row",0,len(df)-1,0)
    base=df.iloc[int(idx)]
    applicant={}
    cols=st.columns(3)
    for i,f in enumerate(bundle["features"]):
        val=base.get(f)
        with cols[i%3]:
            if pd.api.types.is_numeric_dtype(df[f]): applicant[f]=st.number_input(f,value=float(val) if pd.notna(val) else 0.0)
            else:
                choices=df[f].dropna().astype(str).value_counts().head(30).index.tolist()
                current=str(val) if pd.notna(val) else (choices[0] if choices else "Unknown")
                applicant[f]=st.selectbox(f,choices or [current],index=choices.index(current) if current in choices else 0)
    if st.button("Calculate risk",type="primary"):
        st.session_state.prediction=predict_one(bundle,applicant)
    if "prediction" in st.session_state:
        p=st.session_state.prediction
        a,b,c=st.columns(3); a.metric("Default probability",f"{p['default_probability']:.1%}"); b.metric("Risk score",p["risk_score"]); c.metric("Risk band",p["risk_band"])
        st.info("Decision support only: the score must be reviewed with policy, affordability and fair-lending checks.")

with tabs[2]:
    if "prediction" not in st.session_state: st.info("Run a prediction first.")
    else:
        explanation=explain_linear_prediction(bundle,st.session_state.prediction["row"])
        explanation["direction"]=explanation.contribution.apply(lambda x:"Increases risk" if x>0 else "Reduces risk")
        st.plotly_chart(px.bar(explanation,x="contribution",y="feature",orientation="h",color="direction",color_discrete_map={"Increases risk":"#dc2626","Reduces risk":"#0f766e"},title="Largest local SHAP contributions"),use_container_width=True)
        st.caption("Contribution signs explain this prediction relative to the model baseline. They are not causal claims.")
    st.subheader("Business-readable surrogate rules")
    st.code(bundle["rules"])
    st.caption("Rules summarize model behaviour and require policy-owner validation before operational use.")

with tabs[3]:
    st.write("Ask a business question. SQL is generated, validated as read-only, limited, and then executed locally.")
    st.caption("Examples: " + " · ".join(EXAMPLES.keys()))
    if "chat" not in st.session_state: st.session_state.chat=[]
    q=st.chat_input("What is the default rate by gender?")
    if q:
        try:
            sql,result=ask(df,q,st.session_state.chat)
            st.session_state.chat.append({"question":q,"sql":sql,"answer":result.to_dict("records")[:10]})
        except Exception as e: st.error(str(e))
    for item in reversed(st.session_state.chat):
        with st.chat_message("user"): st.write(item["question"])
        with st.chat_message("assistant"):
            st.dataframe(pd.DataFrame(item["answer"]),use_container_width=True)
            with st.expander("Validated SQL"): st.code(item["sql"],language="sql")

with tabs[4]:
    m=bundle["metrics"]
    a,b,c=st.columns(3); a.metric("ROC-AUC",f"{m['roc_auc']:.3f}"); b.metric("PR-AUC",f"{m['pr_auc']:.3f}"); c.metric("Test rows",f"{m['test_rows']:,}")
    st.write("Class imbalance is addressed using stratified splitting and class-weighted logistic regression. PR-AUC is reported alongside ROC-AUC because defaults are the minority class.")
    st.json(m["confusion_matrix"])
