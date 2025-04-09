import pandas as pd

def load_and_clean_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df["unemployment"] = df["unemployment"].astype(str).str.replace(",", ".").astype(float)
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    df["time_index"] = df["year"] * 12 + df["month"]
    return df

def calculate_same_month_avg(df):
    df["same_month_avg"] = 0.0
    for cat in df["category"].unique():
        for month in df["month"].unique():
            mask = (df["category"] == cat) & (df["month"] == month)
            for i, row in df[mask].iterrows():
                past = df[(df["category"] == cat) & (df["month"] == month) & (df["year"] < row["year"])]
                df.at[i, "same_month_avg"] = past["number"].mean() if not past.empty else 0.0
