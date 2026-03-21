import pandas as pd

# 🧠 INTELLIGENT CLUSTER LABELING FUNCTION
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
            label = "High-value inactive"
            behavior = "High spend, low activity"
            risk = "🔴 High churn"
            emoji = "🔴"
            recommend = [
                "Offer personalized discounts",
                "Conduct exit interviews",
                "Assign dedicated account managers"
            ]
        elif low_monetary:
            # CLUSTER 1: LOW MONETARY VALUE - NOT WORTH RETENTION INVESTMENT
            label = "Low engagement"
            behavior = "Low usage"
            risk = "🟡 Medium"
            emoji = "🟡"
            recommend = [
                "Push notifications / reminders",
                "Test re-engagement campaigns",
                "Monitor for account sunset"
            ]
        else:
            # CLUSTER 2: MODERATE VALUE, STABLE - CORE CUSTOMER BASE
            label = "Active loyal"
            behavior = "High usage"
            risk = "🟢 Low"
            emoji = "🟢"
            recommend = [
                "Reward loyalty programs",
                "Implement upsell/cross-sell strategies",
                "Create tiered benefits program"
            ]

        insights[cluster] = {
            "label": label,
            "behavior": behavior,
            "risk": risk,
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

def generate_insights(cluster_insights, feat_imp_df=None):
    """Generate automatic business insights"""

    insights = []

    # Find highest and lowest churn personas
    sorted_insights = sorted(cluster_insights.items(), key=lambda x: x[1]['churn_rate'], reverse=True)
    highest_churn = sorted_insights[0][1]
    lowest_churn = sorted_insights[-1][1]

    insights.append(f"🔴 **{highest_churn['label']}** show the highest churn risk at {highest_churn['stats']['Churn Rate']} - {highest_churn['behavior'].lower()} indicates critical attention needed.")
    insights.append(f"🟢 **{lowest_churn['label']}** demonstrate stability with only {lowest_churn['stats']['Churn Rate']} churn - {lowest_churn['behavior'].lower()} drives loyalty.")

    # Compare engagement vs spending
    high_value_inactive = next((ins for ins in cluster_insights.values() if ins['label'] == 'High-value inactive'), None)
    low_engagement = next((ins for ins in cluster_insights.values() if ins['label'] == 'Low engagement'), None)

    if high_value_inactive and low_engagement:
        if high_value_inactive['churn_rate'] > low_engagement['churn_rate']:
            insights.append("💡 **Engagement trumps spending**: High-value customers with low activity churn more than low-value customers, suggesting poor experience drives churn over pricing.")
        else:
            insights.append("💰 **Value matters**: Even engaged customers with low spending show retention challenges, indicating pricing strategy needs review.")

    # Feature importance insights
    if feat_imp_df is not None and not feat_imp_df.empty:
        top_features = feat_imp_df.head(3)['Feature'].tolist()
        insights.append(f"🔍 **Top churn drivers**: {', '.join(top_features)} are the strongest predictors of customer churn.")
        if 'frequency_log' in top_features or 'purchase_rate' in top_features:
            insights.append("📉 **Activity is key**: Low engagement metrics strongly predict churn - focus on usage improvement initiatives.")

    return insights