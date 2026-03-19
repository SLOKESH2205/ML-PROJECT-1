import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from src.pipeline.predict_pipeline import PredictPipeline

# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# CACHING
# =========================
@st.cache_resource
def load_pipeline():
    return PredictPipeline()

@st.cache_data
def compute_shap_values(_model, X_values):
    import shap
    explainer = shap.TreeExplainer(_model)
    return explainer.shap_values(X_values)

# 🧠 INTELLIGENT CLUSTER LABELING FUNCTION
@st.cache_data
def get_cluster_insights(result_df):
    """Automatically label clusters and generate recommendations"""
    
    # Use raw features if available, otherwise use numeric_only
    raw_cols = ["frequency_log", "monetary_log", "purchase_rate", "Churn_Label", "Churn Probability"]
    available_cols = [col for col in raw_cols if col in result_df.columns]
    
    if available_cols:
        cluster_profile = result_df.groupby("cluster")[available_cols].mean()
    else:
        cluster_profile = result_df.groupby("cluster").mean(numeric_only=True)
    
    overall_means = result_df[available_cols].mean() if available_cols else result_df.mean(numeric_only=True)
    
    insights = {}
    
    for cluster in sorted(result_df["cluster"].unique()):
        cluster_data = cluster_profile.loc[cluster]
        
        # Calculate relative metrics using available columns
        avg_churn = cluster_data.get("Churn Probability", cluster_data.get("Churn_Label", 0))
        avg_frequency = cluster_data.get("frequency_log", 0)
        avg_monetary = cluster_data.get("monetary_log", 0)
        avg_purchase_rate = cluster_data.get("purchase_rate", 0)
        
        overall_churn = overall_means.get("Churn Probability", overall_means.get("Churn_Label", 0))
        overall_frequency = overall_means.get("frequency_log", 0)
        overall_monetary = overall_means.get("monetary_log", 0)
        
        size = len(result_df[result_df["cluster"] == cluster])
        pct = size / len(result_df) * 100
        
        # 🔥 CLUSTER TYPE LOGIC (Explicit, no overlap)
        
        # Define thresholds
        high_churn = avg_churn > overall_churn * 1.3
        high_monetary = avg_monetary > overall_monetary * 1.15
        high_frequency = avg_frequency > overall_frequency * 1.15
        high_value = high_monetary or high_frequency
        
        # Key metric: Is monetary value low?
        low_monetary = avg_monetary < overall_monetary * 0.85
        
        # Determine cluster type - MUTUALLY EXCLUSIVE
        if high_churn and high_value:
            # CLUSTER 0: HIGH-VALUE BUT CHURNING - CRITICAL SEGMENT
            label = "🔴 High-Value At-Risk Customers"
            emoji = "🔴"
            recommend = [
                "⚠️ CRITICAL: Revenue at risk - highest spenders leaving",
                "Conduct exit interviews to identify churn drivers",
                "Offer premium retention packages (loyalty discounts, exclusive features)",
                "Assign dedicated account managers",
                "Create VIP re-engagement campaigns immediately"
            ]
        elif low_monetary:
            # CLUSTER 1: LOW MONETARY VALUE - NOT WORTH RETENTION INVESTMENT
            label = "🟡 Low-Value / Low Engagement Customers"
            emoji = "🟡"
            recommend = [
                "Evaluate CAC vs LTV - retention cost may exceed customer value",
                "Consider offering discounts only if automated/low-cost",
                "Test re-engagement with minimal marketing spend",
                "Monitor for account sunset or pause",
                "Focus resources on Clusters 0 and 2 instead"
            ]
        else:
            # CLUSTER 2: MODERATE VALUE, STABLE - CORE CUSTOMER BASE
            label = "🟢 Core Customer Base"
            emoji = "🟢"
            emoji = "🟢"
            recommend = [
                "Primary revenue driver - protect and grow",
                "Implement upsell/cross-sell strategies to increase AOV",
                "Regular engagement to maintain loyalty",
                "Create tiered benefits program to deepen relationship",
                "Use as reference/advocate base for marketing"
            ]
        
        insights[cluster] = {
            "label": label,
            "emoji": emoji,
            "size": size,
            "percentage": pct,
            "churn_rate": avg_churn * 100,
            "recommendations": recommend,
            "stats": {
                "Avg Frequency": f"{avg_frequency:.2f}",
                "Avg Monetary": f"{avg_monetary:.2f}",
                "Avg Purchase Rate": f"{avg_purchase_rate:.3f}",
                "Churn Rate": f"{avg_churn*100:.1f}%"
            }
        }
    
    return insights

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Settings")

    confidence_threshold = st.slider(
        "Churn Probability Threshold",
        0.0, 1.0, 0.35, 0.05
    )

    show_advanced = st.checkbox("Show Advanced Options")

    st.markdown("---")
    st.markdown("### 📋 Model Info")
    st.write("**Model:** XGBoost")
    st.write("**F1 Score:** ~0.71")

# =========================
# MAIN UI
# =========================
st.title("📊 Customer Churn Prediction App")
st.write("Upload customer data and predict churn probability")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📁 Uploaded Data")
        st.dataframe(df.head())

        st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")

        if st.button("🔍 Predict Churn"):

            pipe = load_pipeline()
            result_df = pipe.predict(df)

            # Apply threshold
            result_df["Churn_Label"] = (
                result_df["Churn Probability"] >= confidence_threshold
            ).astype(int)

            # =========================
            # RESULTS
            # =========================
            st.subheader("📊 Predictions")
            st.dataframe(result_df.head(20))

            total = len(result_df)
            churn_count = result_df["Churn_Label"].sum()
            churn_rate = churn_count / total * 100

            st.metric("Total Customers", total)
            st.metric("Predicted Churn", churn_count)
            st.metric("Churn Rate", f"{churn_rate:.2f}%")

            st.write(
                f"💡 Around {churn_rate:.1f}% customers are at risk (threshold={confidence_threshold})"
            )

            # =========================
            # DISTRIBUTION
            # =========================
            st.subheader("📊 Churn Distribution")

            churn_dist = result_df["Churn_Label"].value_counts()
            st.bar_chart(churn_dist)

            # =========================
            # CUSTOMER SEGMENTS
            # =========================
            st.subheader("🧠 Customer Segments (Clusters)")

            if "cluster" in result_df.columns:
                
                cluster_counts = result_df["cluster"].value_counts().reset_index()
                cluster_counts.columns = ["Cluster", "Customers"]
                cluster_counts["Cluster"] = cluster_counts["Cluster"].apply(lambda x: f"Cluster {x}")
                
                st.dataframe(cluster_counts, use_container_width=True)
                st.bar_chart(cluster_counts.set_index("Cluster"))
                
                # 🔥 CLUSTER PROFILING - UNDERSTAND WHAT EACH CLUSTER MEANS
                st.subheader("📊 Cluster Profile (Mean Characteristics)")
                
                # Use only raw feature columns for profiling
                profile_cols = [col for col in result_df.columns if col in [
                    "frequency_log", "monetary_log", "tenure", 
                    "avg_order_value", "purchase_rate", "Churn Probability"
                ]]
                
                if profile_cols:
                    cluster_profile = result_df.groupby("cluster")[profile_cols].mean()
                    st.dataframe(cluster_profile.round(3), use_container_width=True)
                else:
                    cluster_profile = result_df.groupby("cluster").mean(numeric_only=True)
                    st.dataframe(cluster_profile, use_container_width=True)
                
                st.write("💡 **Interpret the clusters:**")
                st.write("- **Higher Churn Probability** → riskier segment")
                st.write("- **Churn_Label = 1** → predicted churn customers")
                st.write("- Compare values across clusters to understand behavioral differences")

                # 🔥 INTELLIGENT CLUSTER LABELING & RECOMMENDATIONS
                st.subheader("🎯 Cluster Insights & Recommendations")
                
                cluster_insights = get_cluster_insights(result_df)
                
                for cluster_id, insights in sorted(cluster_insights.items()):
                    with st.expander(f"{insights['emoji']} {insights['label']} — {insights['size']} customers ({insights['percentage']:.1f}%)", expanded=True):
                        
                        # Display key statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Avg Frequency", insights['stats']['Avg Frequency'])
                        with col2:
                            st.metric("Avg Monetary", insights['stats']['Avg Monetary'])
                        with col3:
                            st.metric("Purchase Rate", insights['stats']['Avg Purchase Rate'])
                        with col4:
                            st.metric("Churn Rate", insights['stats']['Churn Rate'])
                        
                        # Display recommendations
                        st.write("**🎯 Recommended Actions:**")
                        for i, rec in enumerate(insights['recommendations'], 1):
                            st.write(f"{i}. {rec}")

            else:
                st.warning("⚠️ Cluster information not available in dataset.")

            # =========================
            # HIGH RISK
            # =========================
            st.subheader("⚠️ High Risk Customers")

            high_risk = result_df[
                result_df["Churn Probability"] >= confidence_threshold
            ].sort_values("Churn Probability", ascending=False)

            if len(high_risk) > 0:
                st.dataframe(high_risk.head(10))

                st.download_button(
                    "Download High Risk Customers",
                    high_risk.to_csv(index=False),
                    file_name="high_risk.csv"
                )
            else:
                st.success("No high-risk customers")

            # =========================
            # ADVANCED + SHAP
            # =========================
            if show_advanced:

                st.subheader("📊 Advanced Analytics")

                st.write("Min:", result_df["Churn Probability"].min())
                st.write("Max:", result_df["Churn Probability"].max())
                st.write("Mean:", result_df["Churn Probability"].mean())

                # =========================
                # SHAP
                # =========================
                st.subheader("🔍 Model Explainability (SHAP)")

                import shap
                import matplotlib.pyplot as plt

                try:
                    explainer = shap.TreeExplainer(pipe.model)
                    shap_values = explainer.shap_values(pipe.last_processed_df.values)

                    fig, ax = plt.subplots(figsize=(10, 6))

                    shap.summary_plot(
                        shap_values,
                        pipe.last_processed_df.values,
                        feature_names=pipe.features,
                        plot_type="bar",
                        show=False
                    )

                    st.pyplot(fig)
                    plt.close()
                except Exception as shap_error:
                    st.error(f"⚠️ SHAP Error: {str(shap_error)}")

    except Exception as e:
        st.error(str(e))