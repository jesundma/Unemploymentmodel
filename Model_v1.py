import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from datetime import datetime
import streamlit as st

st.set_page_config(layout="wide")
st.title("📊 Participant Forecasting Dashboard")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Clean data
    df["unemployment"] = df["unemployment"].astype(str).str.replace(",", ".").astype(float)
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    df["time_index"] = df["year"] * 12 + df["month"]

    # Calculate same_month_avg
    df["same_month_avg"] = 0.0
    for cat in df["category"].unique():
        for month in df["month"].unique():
            mask = (df["category"] == cat) & (df["month"] == month)
            for i, row in df[mask].iterrows():
                past = df[(df["category"] == cat) & (df["month"] == month) & (df["year"] < row["year"])]
                df.at[i, "same_month_avg"] = past["number"].mean() if not past.empty else 0.0

    # Forecast settings
    forecast_year = st.slider("Forecast Year", 2025, 2030, 2025)
    forecast_months = st.multiselect("Select Months to Forecast", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    st.subheader("📉 Monthly Unemployment Rate (for Forecast)")
    future_unemployment = {}
    for month in forecast_months:
        future_unemployment[month] = st.number_input(f"Enter forecasted unemployment for month {month}", min_value=0.0, max_value=100.0, value=6.5, step=0.1, key=f"unemp_{month}")

    # Run Models
    rf_results = []
    prophet_results = []
    for cat in df["category"].unique():
        df_cat = df[df["category"] == cat].dropna(subset=["number"])

        # Random Forest
        X = df_cat[["time_index", "unemployment", "same_month_avg"]]
        y = df_cat["number"]
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        for month in forecast_months:
            time_index = forecast_year * 12 + month
            past = df[(df["category"] == cat) & (df["month"] == month) & (df["year"] < forecast_year)]
            same_month_avg = past["number"].mean() if not past.empty else 0.0
            future_unemp = future_unemployment[month]

            pred = rf_model.predict([[time_index, future_unemp, same_month_avg]])[0]
            rf_results.append({"year": forecast_year, "month": month, "category": cat, "forecast_rf": round(pred, 1)})

        # Prophet
        df_prophet = df_cat[["date", "number", "unemployment"]].rename(columns={"date": "ds", "number": "y"})
        prophet_model = Prophet(interval_width=0.95)
        prophet_model.add_regressor("unemployment")
        prophet_model.fit(df_prophet)

        future_dates = pd.date_range(start=f"{forecast_year}-01-01", periods=12, freq="MS")
        future_df = pd.DataFrame({"ds": future_dates})
        future_df["month"] = future_df["ds"].dt.month
        future_df = future_df[future_df["month"].isin(forecast_months)]
        future_df["unemployment"] = future_df["month"].map(future_unemployment)

        forecast = prophet_model.predict(future_df)
        forecast["month"] = forecast["ds"].dt.month
        forecast["year"] = forecast["ds"].dt.year
        forecast["category"] = cat

        # Ensure forecasts are not negative
        forecast["yhat"] = forecast["yhat"].clip(lower=0)  # Set negative forecasts to zero
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)  # Set lower bounds to zero
        forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)  # Set upper bounds to zero

        prophet_results.append(forecast[["year", "month", "category", "yhat", "yhat_lower", "yhat_upper"]])

    df_rf = pd.DataFrame(rf_results)

    df_prophet = pd.concat(prophet_results)
    df_prophet = df_prophet.rename(columns={"yhat": "forecast_prophet", "yhat_lower": "lower_bound", "yhat_upper": "upper_bound"})

    # 📈 Visualizations
    st.subheader("📊 Forecast Results")
    selected_cat = st.selectbox("Select Category for Visualization", sorted(df["category"].unique()))

    rf_chart = df_rf[df_rf["category"] == selected_cat]
    prophet_chart = df_prophet[df_prophet["category"] == selected_cat]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=rf_chart, x="month", y="forecast_rf", label="Random Forest", marker="o")
    sns.lineplot(data=prophet_chart, x="month", y="forecast_prophet", label="Prophet", marker="s")
    ax.fill_between(prophet_chart["month"], prophet_chart["lower_bound"], prophet_chart["upper_bound"], alpha=0.2, label="95% CI")
    ax.set_title(f"Forecast Comparison for Category {selected_cat}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Participants")
    ax.legend()
    st.pyplot(fig)

    # 📋 Summary Statistics
    st.subheader("📋 Forecast Summary Table")
    summary_table = pd.merge(df_rf, df_prophet, on=["year", "month", "category"])
    st.dataframe(summary_table)

    # 📤 Download Results
    st.subheader("📤 Download Forecast Results")
    output = pd.ExcelWriter("forecast_output.xlsx", engine='xlsxwriter')
    df_rf.to_excel(output, sheet_name="RandomForest", index=False)
    df_prophet.to_excel(output, sheet_name="Prophet", index=False)
    output.close()

    with open("forecast_output.xlsx", "rb") as f:
        st.download_button("Download Excel File", f, file_name="forecast_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")