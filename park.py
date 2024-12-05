import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Set the title of the Streamlit app
st.title("Parking Availability in Bologna")

# Define the dataset API URL
DATASET_API_URL = "https://opendata.comune.bologna.it/api/records/1.0/search/"

# Fetch data from the API
@st.cache_data
def fetch_parking_data():
    params = {
        "dataset": "disponibilita-parcheggi-storico",
        "sort": "data",
        "rows": 100,  # You can adjust the number of rows
    }
    response = requests.get(DATASET_API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch data. Please try again later.")
        return None

# Process and load data
def process_data(data):
    records = data.get("records", [])
    processed_data = []
    for record in records:
        fields = record.get("fields", {})
        processed_data.append({
            "Parking Lot": fields.get("nome"),
            "Available Spaces": fields.get("posti_disponibili"),
            "Total Spaces": fields.get("capienza"),
            "Date": fields.get("data"),
            "Coordinates": fields.get("geolocation"),
        })
    return pd.DataFrame(processed_data)

# Fetch and display data
data = fetch_parking_data()
if data:
    df = process_data(data)

    # Map Visualization
    st.header("Parking Availability Map")
    if not df.empty and "Coordinates" in df.columns:
        df["Latitude"] = df["Coordinates"].apply(lambda x: x[0] if x else None)
        df["Longitude"] = df["Coordinates"].apply(lambda x: x[1] if x else None)

        fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            hover_name="Parking Lot",
            hover_data=["Available Spaces", "Total Spaces"],
            color="Available Spaces",
            size="Available Spaces",
            color_continuous_scale="Viridis",
            mapbox_style="open-street-map",
            title="Real-Time Parking Availability"
        )
        st.plotly_chart(fig)

    # Data Table
    st.header("Parking Data")
    st.dataframe(df)

    # Filter Data
    st.sidebar.header("Filter Options")
    selected_lot = st.sidebar.selectbox("Select a Parking Lot", df["Parking Lot"].unique())
    filtered_df = df[df["Parking Lot"] == selected_lot]
    st.subheader(f"Details for {selected_lot}")
    st.dataframe(filtered_df)
else:
    st.write("No data available.")

