import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# Load model and dataset
# =============================
model = joblib.load("attrition_model.pkl")
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Preprocess dataset for analysis
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# =============================
# Streamlit setup
# =============================
st.set_page_config(page_title="AI Attrition Agent", layout="wide")
st.title("💼 HR Attrition Prediction & Analytics Agent")

tab1, tab2 = st.tabs(["🔮 Predict Attrition Risk", "💬 Ask the AI Agent"])

# =============================
# TAB 1 — PREDICTION
# =============================
with tab1:
    st.sidebar.header("Employee Profile")

    age = st.sidebar.slider("Age", 18, 60, 30)
    monthly_income = st.sidebar.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
    overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
    job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
    worklife_balance = st.sidebar.slider("Work-Life Balance (1-4)", 1, 4, 3)
    years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)

    input_data = pd.DataFrame({
        'Age': [age],
        'MonthlyIncome': [monthly_income],
        'OverTime': [1 if overtime == "Yes" else 0],
        'JobSatisfaction': [job_satisfaction],
        'WorkLifeBalance': [worklife_balance],
        'YearsAtCompany': [years_at_company]
    })

    st.subheader("🔍 Attrition Risk Prediction")
    if st.button("Predict Attrition Risk"):
        prob = model.predict_proba(input_data)[0][1]
        risk = "High" if prob > 0.65 else "Medium" if prob > 0.4 else "Low"

        st.subheader(f"🧭 Predicted Attrition Risk: **{risk}** ({prob:.2f})")

        if risk == "High":
            st.warning("**High Risk:** Employee likely to leave. Recommended Actions:")
            st.markdown("- Reduce overtime and improve work-life balance.")
            st.markdown("- Offer recognition or financial incentives.")
            st.markdown("- Provide career growth opportunities.")
        elif risk == "Medium":
            st.info("**Medium Risk:** Moderate likelihood of leaving. Recommended Actions:")
            st.markdown("- Monitor job satisfaction and engagement.")
            st.markdown("- Review compensation and workload.")
        else:
            st.success("**Low Risk:** Employee stable. Recommended Actions:")
            st.markdown("- Maintain positive work environment and recognition programs.")

        st.progress(int(prob * 100))

# =============================
# TAB 2 — CONVERSATIONAL ANALYTICS
# =============================
with tab2:
    st.subheader("💬 Ask the AI Agent about HR Analytics")

    st.markdown("You can ask questions like:")
    st.markdown("- 'Which department has highest attrition?'")
    st.markdown("- 'How does overtime affect attrition?'")
    st.markdown("- 'Show attrition by age group.'")
    st.markdown("- 'Average income of employees who left.'")

    query = st.chat_input("Ask a question about the dataset...")

    if query:
        st.chat_message("user").write(query)
        response = ""
        q = query.lower()

        if "department" in q:
            attr_by_dept = df.groupby("Department")["Attrition"].mean().sort_values(ascending=False)
            st.bar_chart(attr_by_dept)
            response = f"Attrition is highest in **{attr_by_dept.index[0]}** department."
        
        elif "overtime" in q:
            attr_by_ot = df.groupby("OverTime")["Attrition"].mean()
            st.bar_chart(attr_by_ot)
            response = "Employees working overtime have a significantly higher attrition rate."
        
        elif "age" in q:
            df["AgeGroup"] = pd.cut(df["Age"], bins=[18,25,35,45,60], labels=["18-25","26-35","36-45","46-60"])
            attr_by_age = df.groupby("AgeGroup")["Attrition"].mean()
            st.line_chart(attr_by_age)
            response = "Attrition tends to be higher among younger employees (18–35)."
        
        elif "income" in q or "salary" in q:
            fig, ax = plt.subplots()
            sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, ax=ax)
            st.pyplot(fig)
            response = "Lower income groups show higher attrition tendency."
        
        elif "education" in q:
            attr_by_edu = df.groupby("EducationField")["Attrition"].mean()
            st.bar_chart(attr_by_edu)
            response = "Attrition varies slightly by education field, highest in HR and Sales."
        
        elif "job" in q or "role" in q:
            attr_by_role = df.groupby("JobRole")["Attrition"].mean().sort_values(ascending=False)
            st.bar_chart(attr_by_role)
            response = f"Top attrition roles: {', '.join(attr_by_role.head(3).index)}"
        
        elif "reason" in q or "factor" in q or "why" in q:
            response = "Top influencing factors are OverTime, Monthly Income, and Job Satisfaction."

        else:
            response = "I can answer HR-related questions about attrition, overtime, age, income, or departments."

        st.chat_message("assistant").write(response)

st.caption("Developed by [Your Name] — HR Analytics Project (Streamlit Cloud)")