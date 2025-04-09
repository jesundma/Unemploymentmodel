import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_forecast_results(rf_results, prophet_results):
    df_rf = pd.DataFrame(rf_results)
    df_prophet = pd.concat(prophet_results, ignore_index=True)

    st.subheader("📊 Forecast Results")
    selected_cat = st.selectbox("Select Category for Visualization", sorted(df_rf["category"].unique()))

    rf_chart = df_rf[df_rf["category"] == selected_cat]
    prophet_chart = df_prophet[df_prophet["category"] == selected_cat]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=rf_chart, x="month", y="forecast_rf", label="Random Forest", marker="o")
    sns.lineplot(data=prophet_chart, x="month", y="forecast_prophet", label="Prophet", marker="s")
    ax.fill_between(prophet_chart["month"], prophet_chart["yhat_lower"], prophet_chart["yhat_upper"], alpha=0.2, label="95% CI")
    ax.set_title(f"Forecast Comparison for Category {selected_cat}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Participants")
    ax.legend()
    st.pyplot(fig)

    # 📋 Summary Statistics
    st.subheader("📋 Forecast Summary Table")
    summary_table = pd.merge(df_rf, df_prophet, on=["year", "month", "category"])
    st.dataframe(summary_table)
