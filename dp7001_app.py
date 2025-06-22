# dp7001_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# Load data
df = pd.read_csv("result.csv")

# Define numeric features for clustering
features = ['age', 'years_of_experience', 'hours_worked_per_week', 'number_of_virtual_meetings',
            'work_life_balance_rating', 'social_isolation_rating', 'company_support_for_remote_work']

# Apply custom style
st.set_page_config(page_title="Mental Health Clustering Dashboard", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f9fbfc; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Cluster Overview", "Predict Your Cluster"],
                           icons=['bar-chart-line', 'person-bounding-box'], menu_icon="cast", default_index=0)

if selected == "Cluster Overview":
    st.title("\U0001F4CA Cluster Distribution Overview")

    st.subheader("\U0001F4DD Cluster Descriptions")
    st.markdown("""
    - **Cluster 0:** Older employees with higher experience and strong company support. They tend to work longer hours and face moderate social isolation.
    - **Cluster 1:** Mid-career individuals with balanced work-life ratings and moderate stress levels.
    - **Cluster 2:** Younger employees, often new to the workforce, with lower support and work-life satisfaction.
    """)

    st.subheader("\U0001F4D1 Categorical Distribution by Cluster")
    cat_vars = ['gender', 'stress_level', 'mental_health_condition', 'satisfaction_with_remote_work']
    for cat in cat_vars:
        fig = px.histogram(df, x=cat, color='cluster', barmode='group', title=f"{cat.replace('_', ' ').title()} by Cluster")
        st.plotly_chart(fig, use_container_width=True)

    for col in features:
        fig = px.box(df, x='cluster', y=col, color='cluster', title=f"{col.replace('_', ' ').title()} by Cluster")
        st.plotly_chart(fig, use_container_width=True)

elif selected == "Predict Your Cluster":
    st.title("\U0001F916 Predict Your Cluster")

    st.markdown("""
    Fill out the form below with your personal data. Once all fields are completed, click **Predict** to find out which mental health support cluster you fall into.
    
    **Scale Reference for Ratings:**
    - 1: Very Low / Not at all
    - 2: Low
    - 3: Moderate / Neutral
    - 4: High
    - 5: Very High / Extreme
    """)

    # Initialize or reset session state
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            'age': None,
            'exp': None,
            'hours': None,
            'meetings': None,
            'wlbr': 1,
            'sir': 1,
            'csr': 1
        }
        st.session_state.pred_cluster = None
        st.session_state.submitted = False

    if st.button("🔄 Refresh Form"):
        st.session_state.form_data = {
            'age': None,
            'exp': None,
            'hours': None,
            'meetings': None,
            'wlbr': 1,
            'sir': 1,
            'csr': 1
        }
        st.session_state.pred_cluster = None
        st.session_state.submitted = False
        st.rerun()

    with st.form("cluster_form"):
        form_data = st.session_state.form_data

        form_data['age'] = st.number_input("Age", min_value=18, max_value=70, value=form_data['age'])
        form_data['exp'] = st.number_input("Years of Experience", min_value=0, max_value=50, value=form_data['exp'])
        form_data['hours'] = st.number_input("Hours Worked Per Week", min_value=1, max_value=100, value=form_data['hours'])
        form_data['meetings'] = st.number_input("Number of Virtual Meetings", min_value=0, max_value=20, value=form_data['meetings'])

        form_data['wlbr'] = st.slider("Work-Life Balance Rating", 1, 5, value=form_data['wlbr'],
                                      help="How well you can balance your work and personal life")
        form_data['sir'] = st.slider("Social Isolation Rating", 1, 5, value=form_data['sir'],
                                     help="How isolated you feel from colleagues or friends")
        form_data['csr'] = st.slider("Company Support for Remote Work", 1, 5, value=form_data['csr'],
                                     help="How much support your company provides for remote work")

        submitted = st.form_submit_button("Predict")

    if submitted:
        if None in [form_data['age'], form_data['exp'], form_data['hours'], form_data['meetings']]:
            st.warning("Please complete all fields before predicting.")
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(df[features])
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(X)

            user_data = pd.DataFrame([[form_data['age'], form_data['exp'], form_data['hours'], form_data['meetings'],
                                        form_data['wlbr'], form_data['sir'], form_data['csr']]], columns=features)
            user_scaled = scaler.transform(user_data)
            pred_cluster = kmeans.predict(user_scaled)[0]

            st.session_state.pred_cluster = pred_cluster
            st.session_state.submitted = True

    if st.session_state.submitted and st.session_state.pred_cluster is not None:
        pred_cluster = st.session_state.pred_cluster
        st.success(f"✅ You belong to **Cluster {pred_cluster}**")

        cluster_map = {
            0: "Older employees with more experience, strong company support, and higher work hours.",
            1: "Balanced professionals with stable work-life and stress levels.",
            2: "Younger or early-career employees with lower support and greater challenges managing work-life."
        }
        st.markdown(f"**Description:** {cluster_map[pred_cluster]}")
    else:
        st.info("Submit the form to see your cluster.")