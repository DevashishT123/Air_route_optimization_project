import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium import Marker, Icon, PolyLine

# Load data
hub_df = pd.read_csv("Hub_coordinates - Sheet1.csv")
airport_df = pd.read_csv("airport_coordinates.csv")

# Helper: extract city from hub name (e.g., "A01-AHMEDABAD APEX" -> "AHMEDABAD")
def extract_city(hub_name):
    # Remove code and "APEX", keep city part
    parts = hub_name.split('-')
    if len(parts) > 1:
        city = parts[1].replace("APEX", "").strip()
        return city
    return hub_name

# Build mapping: for each hub, find matching airport by city substring
hub_airport_pairs = []
for _, hub in hub_df.iterrows():
    hub_city = extract_city(hub['Hub Name']).upper()
    # Find airport with matching city
    airport_row = airport_df[airport_df['Airport City'].str.upper().str.contains(hub_city)]
    if not airport_row.empty:
        airport = airport_row.iloc[0]
        hub_airport_pairs.append({
            "hub_name": hub['Hub Name'],
            "hub_lat": hub['Latitude'],
            "hub_lon": hub['Longitude'],
            "airport_city": airport['Airport City'],
            "airport_lat": airport['Latitude'],
            "airport_lon": airport['Longitude']
        })

# Create map
m = folium.Map(location=[22, 80], zoom_start=5, tiles='cartodbpositron')

# Add all hubs and airports
for pair in hub_airport_pairs:
    Marker([pair['hub_lat'], pair['hub_lon']], popup=pair['hub_name'], icon=Icon(color='red', icon='info-sign')).add_to(m)
    Marker([pair['airport_lat'], pair['airport_lon']], popup=pair['airport_city'], icon=Icon(color='green', icon='plane')).add_to(m)
    # Draw dashed line
    PolyLine(
        locations=[(pair['hub_lat'], pair['hub_lon']), (pair['airport_lat'], pair['airport_lon'])],
        color='blue',
        weight=2,
        dash_array='10'
    ).add_to(m)

st.title("Hub ↔ Airport Network Graph")
st_folium(m, height=700, width=1000)