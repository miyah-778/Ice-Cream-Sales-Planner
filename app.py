import pickle
from datetime import date, timedelta
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# ---- Settings ----------------------------------------------------------
CITIES = {
    "Pittsburgh": (40.44, -79.99),
    "New York": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Miami": (25.76, -80.19),
    "Los Angeles": (34.05, -118.24),
    "Toronto": (43.65, -79.38),
    "Vancouver": (49.28, -123.12),
}
LOW_TEMP, HIGH_TEMP = 50, 100
LOW_SALES, HIGH_SALES = 200, 300
URL = "https://api.open-meteo.com/v1/forecast"
# ------------------------------------------------------------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Ice Cream Sales Planner", page_icon="🍦", layout="wide")
st.title("🍦 Ice Cream Sales Planner")
st.caption("Use weather forecasts and machine learning to predict sales and plan your inventory for the next 16 days.")

city = st.selectbox("City", list(CITIES))
selected_date = st.date_input("Date", value=date.today() + timedelta(days=1), min_value=date.today(), max_value=date.today() + timedelta(days=15))

if st.button("Predict sales", type="primary"):
    # 1. Get the 16-day weather forecast
    latitude, longitude = CITIES[city]
    try:
        response = requests.get(URL, params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,precipitation_sum",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "mm",
            "timezone": "auto",
            "forecast_days": 16,
        }, timeout=30)
        response.raise_for_status()
        weather = response.json()["daily"]
    except (requests.RequestException, KeyError, ValueError) as e:
        st.error(f"Unable to retrieve the weather forecast: {e}")
        st.stop()

    # 2. Predict sales for all 16 days
    df = pd.DataFrame({
        "date": pd.to_datetime(weather["time"]),
        "temperature": weather["temperature_2m_max"],
        "precipitation": weather["precipitation_sum"],
    })
    df = df.dropna().copy()
    if df.empty:
        st.error("No complete weather forecast is available.")
        st.stop()
    features = df[["temperature", "precipitation"]]
    df["sales"] = model.predict(features).clip(min=0)
    df["day"] = df["date"].dt.strftime("%b %d")
    df["rank"] = df["sales"].rank(ascending=False, method="min").astype(int)

    # 3. Calculate sales benchmarks
    selected = df[df["date"].dt.date == selected_date]
    if selected.empty:
        st.warning("The selected date has no complete forecast. Please choose another date.")
        st.stop()
    selected = selected.iloc[0]
    prediction = selected["sales"]
    max_sales = df["sales"].max()
    avg_sales = df["sales"].mean()
    max_date = df.loc[df["sales"].idxmax(), "date"]
    ratio = prediction / max_sales * 100 if max_sales > 0 else 0

    # 4. Display selected-day metrics
    st.subheader(f"Sales Forecast for {selected_date:%B %d, %Y}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Forecast high", f"{selected['temperature']:.0f} °F")
    col2.metric("Precipitation", f"{selected['precipitation']:.1f} mm")
    col3.metric("Predicted sales", f"{prediction:.0f} cones")
    col4.metric("16-day maximum", f"{max_sales:.0f} cones")

    st.write(f"**Selected day vs. 16-day maximum:** {prediction:.0f} / {max_sales:.0f} cones ({ratio:.0f}%)")
    st.progress(min(ratio / 100, 1.0))
    st.caption(f"16-day average: {avg_sales:.0f} cones/day | Highest sales date: {max_date:%B %d} | Selected day's rank: #{selected['rank']} of {len(df)}")

    # 5. Classify absolute and relative demand
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Overall Demand")
        if prediction > HIGH_SALES:
            st.success("Busy day: make sure you are fully stocked.")
        elif prediction > LOW_SALES:
            st.info("Normal day: maintain regular inventory.")
        else:
            st.warning("Slow day: consider a promotion or order less.")
    with col2:
        st.markdown("#### Relative Demand")
        if selected["rank"] <= 5:
            st.success("Top 5 forecast day: consider preparing extra inventory.")
        elif prediction > avg_sales:
            st.info("Above-average day: expect relatively strong demand.")
        else:
            st.info("Below-average day: standard inventory should be sufficient.")

    if selected["temperature"] < LOW_TEMP or selected["temperature"] > HIGH_TEMP or selected["precipitation"] > 20:
        st.warning("Selected weather conditions are outside the model's training range. Treat the prediction with caution.")

    # 6. Visualize the 16-day forecast
    st.subheader("16-Day Sales Forecast")
    st.caption("Blue = selected date | Orange = maximum sales | Purple = both")
    selected_day = pd.Timestamp(selected_date)
    df["category"] = "Other days"
    df.loc[df["date"] == selected_day, "category"] = "Selected date"
    df.loc[df["sales"] == max_sales, "category"] = "Maximum sales"
    df.loc[(df["date"] == selected_day) & (df["sales"] == max_sales), "category"] = "Selected & maximum"

    fig = px.bar(df, x="day", y="sales", color="category",
        color_discrete_map={
            "Other days": "#B8C2CC",
            "Selected date": "#3498DB",
            "Maximum sales": "#F39C12",
            "Selected & maximum": "#9B59B6",
        },
        hover_data={"temperature": ":.1f", "precipitation": ":.1f", "sales": ":.0f", "rank": True, "category": False},
        labels={"day": "Date", "sales": "Predicted cones", "category": "Day type", "temperature": "Temperature (°F)", "precipitation": "Rainfall (mm)", "rank": "Sales rank"})
    fig.add_hline(y=avg_sales, line_dash="dash", line_color="green", annotation_text="16-day average", annotation_position="top left")
    fig.update_layout(xaxis_title="Date", yaxis_title="Predicted cones", legend_title_text="", bargap=0.2)
    st.plotly_chart(fig, use_container_width=True)

    # 7. Show detailed forecast
    with st.expander("View 16-day forecast data"):
        display_df = df[["day", "temperature", "precipitation", "sales", "rank"]].copy()
        display_df.columns = ["Date", "Temperature (°F)", "Precipitation (mm)", "Predicted cones", "Rank"]
        st.dataframe(display_df.round(1), hide_index=True, use_container_width=True)
