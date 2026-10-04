import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Credit Risk Assessment",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# LOAD ML MODEL
# ============================================================

rf_clean = joblib.load("models/rf_clean.pkl")
encoder_clean = joblib.load("models/encoder_clean.pkl")

explainer_clean = shap.TreeExplainer(rf_clean)


# ============================================================
# MODEL FEATURES
# ============================================================

categorical_cols = [
    "education",
    "self_employed"
]

numerical_cols = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]


# ============================================================
# RAG SETUP
# ============================================================

@st.cache_resource
def create_rag():

    # Read PDF
    reader = PdfReader("credit_policy.pdf")

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Create chunks
    chunk_size = 500

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(
            text[i:i + chunk_size]
        )

    # Create LangChain documents
    documents = [
        Document(page_content=chunk)
        for chunk in chunks
    ]

    # Embedding model
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FAISS vector store
    vectorstore = FAISS.from_documents(
        documents,
        embeddings_model
    )

    return vectorstore


vectorstore = create_rag()


# ============================================================
# RAG RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# ============================================================
# RAG PROMPT
# ============================================================

prompt = PromptTemplate.from_template(
    """
You are a strict credit policy assistant.

Answer the question ONLY if the answer is explicitly stated
in the provided context.

Rules:

1. Do not use outside knowledge.
2. Do not guess or infer.
3. If the context does not explicitly answer the question,
   say exactly:

"I don't know based on the provided credit policy."

4. Give a short and direct answer.
5. Ignore information that does not answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# RAG RELEVANCE CHECK
# ============================================================

def retrieve_relevant_docs(question):

    results = vectorstore.similarity_search_with_score(
        question,
        k=2
    )

    threshold = 1.20

    relevant_docs = [
        doc
        for doc, score in results
        if score <= threshold
    ]

    return relevant_docs


# ============================================================
# RAG ANSWER FUNCTION
# ============================================================

def answer_question(question):

    docs = retrieve_relevant_docs(question)

    if len(docs) == 0:
        return "I don't know based on the provided credit policy."

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt_text = prompt.format(
        context=context,
        question=question
    )

    # Simple extractive fallback.
    # This avoids requiring a separate LLM inside Streamlit.
    return context


# ============================================================
# TITLE
# ============================================================

st.title("💳 AI Credit Risk Assessment System")

st.write(
    "AI-powered loan approval, risk assessment, "
    "explainability, and credit policy assistant."
)


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.subheader("Applicant Information")


col1, col2 = st.columns(2)


with col1:

    no_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=2
    )

    education = st.selectbox(
        "Education",
        [" Graduate", " Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        [" No", " Yes"]
    )

    income_annum = st.number_input(
        "Annual Income",
        min_value=0,
        value=500000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=1500000
    )

    loan_term = st.number_input(
        "Loan Term",
        min_value=1,
        max_value=30,
        value=10
    )


with col2:

    cibil_score = st.number_input(
        "CIBIL Score",
        min_value=300,
        max_value=900,
        value=750
    )

    residential_assets_value = st.number_input(
        "Residential Assets Value",
        min_value=0,
        value=2000000
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets Value",
        min_value=0,
        value=500000
    )

    luxury_assets_value = st.number_input(
        "Luxury Assets Value",
        min_value=0,
        value=300000
    )

    bank_asset_value = st.number_input(
        "Bank Asset Value",
        min_value=0,
        value=1000000
    )


# ============================================================
# CREDIT ASSESSMENT
# ============================================================

if st.button(
    "🔍 Assess Credit Risk",
    use_container_width=True
):

    # --------------------------------------------------------
    # CREATE APPLICANT DATAFRAME
    # --------------------------------------------------------

    applicant = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }

    applicant_df = pd.DataFrame(
        [applicant]
    )


    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    applicant_num = applicant_df[
        numerical_cols
    ]

    applicant_cat = applicant_df[
        categorical_cols
    ]

    applicant_cat_encoded = encoder_clean.transform(
        applicant_cat
    )

    applicant_final = np.hstack(
        [
            applicant_num.values,
            applicant_cat_encoded
        ]
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = rf_clean.predict(
        applicant_final
    )[0]

    approval_probability = rf_clean.predict_proba(
        applicant_final
    )[0, 1]

    risk_score = 1 - approval_probability


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if prediction == 1:
        loan_status = "Approved"
    else:
        loan_status = "Rejected"


    # --------------------------------------------------------
    # RISK CATEGORY
    # --------------------------------------------------------

    if risk_score < 0.30:

        risk_category = "Low Risk"

    elif risk_score < 0.60:

        risk_category = "Medium Risk"

    else:

        risk_category = "High Risk"


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.subheader("📊 Credit Risk Assessment")


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        if loan_status == "Approved":
            st.success(
                f"Loan Status: {loan_status}"
            )
        else:
            st.error(
                f"Loan Status: {loan_status}"
            )


    with result_col2:

        st.metric(
            "Approval Probability",
            f"{approval_probability * 100:.2f}%"
        )


    with result_col3:

        st.metric(
            "Risk Score",
            f"{risk_score * 100:.2f}%"
        )


    st.info(
        f"Risk Category: **{risk_category}**"
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.subheader(
        "🔎 Why did the model make this decision?"
    )


    shap_values = explainer_clean.shap_values(
        applicant_final
    )

    shap_values_approved = shap_values[:, :, 1]


    feature_names = (
        numerical_cols
        + list(
            encoder_clean.get_feature_names_out(
                categorical_cols
            )
        )
    )


    applicant_explanation = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Value": shap_values_approved[0]
        }
    )


    applicant_explanation["Impact"] = (
        applicant_explanation["SHAP Value"].abs()
    )


    applicant_explanation = (
        applicant_explanation
        .sort_values(
            "Impact",
            ascending=False
        )
    )


    for _, row in applicant_explanation.head(5).iterrows():

        if row["SHAP Value"] > 0:

            direction = "supports approval"

        else:

            direction = "increases rejection risk"


        st.write(
            f"**{row['Feature']}** → {direction}"
        )


    # ========================================================
    # CREDIT POLICY ASSISTANT
    # ========================================================

    st.divider()

    st.subheader(
        "📚 Ask the Credit Policy"
    )

    policy_question = st.text_input(
        "Ask a question about the credit policy:",
        placeholder="Example: What are the standard payment terms?"
    )


    if policy_question:

        policy_answer = answer_question(
            policy_question
        )

        st.write(
            policy_answer
        )
