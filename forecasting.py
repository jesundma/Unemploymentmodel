import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

def random_forest_forecast(df, category, forecast_year, forecast_months, future_unemployment):
    df_cat = df[df["category"] == category].dropna(subset=["number"])
    rf_results = []
    X = df_cat[["time_index", "unemployment", "same_month_avg"]]
    y = df_cat["number"]
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    for month in forecast_months:
        time_index = forecast_year * 12 + month
        past = df[(df["category"] == category) & (df["month"] == month) & (df["year"] < forecast_year)]
        same_month_avg = past["number"].mean() if not past.empty else 0.0
        future_unemp = future_unemployment[month]

        pred = rf_model.predict([[time_index, future_unemp, same_month_avg]])[0]
        rf_results.append({"year": forecast_year, "month": month, "category": category, "forecast_rf": round(pred, 1)})
    
    return rf_results

def prophet_forecast(df, category, forecast_year, forecast_months, future_unemployment):
    df_cat = df[df["category"] == category][["date", "number", "unemployment"]].rename(columns={"date": "ds", "number": "y"})
    prophet_model = Prophet(interval_width=0.95)
    prophet_model.add_regressor("unemployment")
    prophet_model.fit(df_cat)

    future_dates = pd.date_range(start=f"{forecast_year}-01-01", periods=12, freq="MS")
    future_df = pd.DataFrame({"ds": future_dates})
    future_df["month"] = future_df["ds"].dt.month
    future_df = future_df[future_df["month"].isin(forecast_months)]
    future_df["unemployment"] = future_df["month"].map(future_unemployment)

    forecast = prophet_model.predict(future_df)
    forecast["month"] = forecast["ds"].dt.month
    forecast["year"] = forecast["ds"].dt.year
    forecast["category"] = category

    # Ensure forecasts are not negative
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    return forecast[["year", "month", "category", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"yhat": "forecast_prophet"})
