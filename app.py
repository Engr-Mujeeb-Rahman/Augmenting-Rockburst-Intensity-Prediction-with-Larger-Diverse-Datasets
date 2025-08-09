import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(page_title="🌋 Rockburst Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("🚧 Rockburst Event Analysis Dashboard")
st.markdown("""
Welcome to the interactive dashboard for **Rockburst Event Analysis**.  
Use the sidebar to filter data and explore various visualizations related to seismic events and rockburst intensity.
""")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("E:/desktop/sss.csv")

df = load_data()

# --- Sidebar ---
st.sidebar.title("🔧 Filters")

rock_types = df['rock_type'].dropna().unique().tolist()
locations = df['location'].dropna().unique().tolist()

selected_rock = st.sidebar.multiselect("Rock Type", rock_types, default=rock_types)
selected_location = st.sidebar.multiselect("Location", locations, default=locations)

df_filtered = df[df['rock_type'].isin(selected_rock) & df['location'].isin(selected_location)]

# --- Tabs Layout ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Dataset", "📊 Statistics", "📈 Visualizations", "🧩 Correlation"
])

# --- Tab 1: Dataset Overview ---
with tab1:
    st.subheader("📄 Filtered Dataset")
    st.dataframe(df_filtered, use_container_width=True)
    st.caption(f"Total Records: {df_filtered.shape[0]}")

# --- Tab 2: Summary Statistics ---
with tab2:
    st.subheader("📊 Descriptive Statistics")
    st.dataframe(df_filtered.describe(), use_container_width=True)

# --- Tab 3: Visualizations ---
with tab3:
    st.subheader("📈 Visual Insights")

    numeric_cols = df_filtered.select_dtypes(include='number').columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis", numeric_cols, index=numeric_cols.index("depth") if "depth" in numeric_cols else 0)
    with col2:
        y_axis = st.selectbox("Y-axis", numeric_cols, index=numeric_cols.index("signal_energy") if "signal_energy" in numeric_cols else 1)

    st.plotly_chart(px.scatter(df_filtered, x=x_axis, y=y_axis, color="intensity_label",
                               title=f"{y_axis} vs {x_axis}",
                               template="plotly_dark"), use_container_width=True)

    hist_feature = st.selectbox("📊 Histogram Feature", numeric_cols, index=0)
    st.plotly_chart(px.histogram(df_filtered, x=hist_feature, color="rock_type", nbins=30,
                                  title=f"Distribution of {hist_feature}",
                                  template="simple_white"), use_container_width=True)

    box_feature = st.selectbox("📦 Boxplot Feature", numeric_cols, index=1)
    st.plotly_chart(px.box(df_filtered, y=box_feature, color="intensity_label", points="outliers",
                            title=f"{box_feature} by Intensity", template="ggplot2"),
                    use_container_width=True)

# --- Tab 4: Correlation Heatmap ---
with tab4:
    st.subheader("🧩 Variable Correlation (Pearson)")

    corr = df_filtered[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    st.pyplot(fig)