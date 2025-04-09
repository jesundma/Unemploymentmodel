import pandas as pd
import streamlit as st
from forecasting import random_forest_forecast, prophet_forecast

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
    df["same_month_avg"] = df.groupby(['category', 'month'])['number'].transform(lambda x: x.shift().mean()).fillna(0)

    # Forecast settings
    forecast_year = st.slider("Forecast Year", 2025, 2030, 2025)
    forecast_months = st.multiselect("Select Months to Forecast", list(range(1, 13)), default=list(range(1, 13)))

    st.subheader("📉 Monthly Unemployment Rate (for Forecast)")
    future_unemployment = {}
    for month in forecast_months:
        future_unemployment[month] = st.number_input(f"Enter forecasted unemployment for month {month}", min_value=0.0, max_value=100.0, value=6.5, step=0.1)

    # Run Models
    rf_results, prophet_results = [], []
    for cat in df["category"].unique():
        rf_results.extend(random_forest_forecast(df, cat, forecast_year, forecast_months, future_unemployment))
        prophet_results.append(prophet_forecast(df, cat, forecast_year, forecast_months, future_unemployment))

    # Visualizations
    from visualization import plot_forecast_results
    plot_forecast_results(rf_results, prophet_results)

    # Download Results
    st.subheader("📤 Download Forecast Results")
    output = pd.ExcelWriter("forecast_output.xlsx", engine='xlsxwriter')
    pd.DataFrame(rf_results).to_excel(output, sheet_name="RandomForest", index=False)
    pd.concat(prophet_results, ignore_index=True).to_excel(output, sheet_name="Prophet", index=False)
    output.close()

    with open("forecast_output.xlsx", "rb") as f:
        st.download_button("Download Excel File", f, file_name="forecast_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
