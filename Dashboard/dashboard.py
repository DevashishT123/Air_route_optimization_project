import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium import Marker, Icon, PolyLine
import plotly.express as px
import plotly.graph_objects as go
from branca.element import Template, MacroElement
import numpy as np
import re
from PIL import Image
import base64
import os

# Page config
st.set_page_config(
    page_title="Logistics Dashboard",
    page_icon="🚛",
    layout="wide"
)

# Load Data Functions
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load datasets
shipment_df = load_data("/Users/devashish/sample/LETS_MAKE_IT_FINAL/final_route_planning_for_each_OD_pair.csv")
hub_coords_df = load_data("/Users/devashish/sample/LETS_MAKE_IT_FINAL/Hub_coordinates - Sheet1.csv")
branch_coords_df = load_data("/Users/devashish/sample/LETS_MAKE_IT_FINAL/branch_coordinates_new.csv")
airport_coords_df = load_data("/Users/devashish/sample/LETS_MAKE_IT_FINAL/airport_coordinates.csv")
flight_connections_df = load_data("/Users/devashish/sample/LETS_MAKE_IT_FINAL/flight_connection(sample).csv")
calc_volume_df = load_data("/Users/devashish/sample/LETS_MAKE_IT_FINAL/Total_volume_from_airport_to_air(corrected).csv")
actual_flow_df = load_data("/Users/devashish/sample/LETS_MAKE_IT_FINAL/Airport_to_airport_volume(Actual).csv")
updated_volume_df = load_data("/Users/devashish/sample/LETS_MAKE_IT_FINAL/updated_airport_to_airport_volume.csv")

# Helper Functions
def get_coordinates(location, dataframes):
    """Get coordinates for a location from multiple dataframes"""
    for df in dataframes:
        name_col = df.columns[df.columns.str.contains('Name|City', case=False)][0]
        # Ensure string type for .str.contains
        match = df[df[name_col].astype(str).str.contains(str(location), case=False, na=False)]
        if not match.empty:
            return match.iloc[0]['Latitude'], match.iloc[0]['Longitude']
    return None, None

def parse_route(route_str):
    """Parse route string into segments"""
    segments = []
    nodes = route_str.split('→')
    nodes = [node.strip() for node in nodes]
    
    for i in range(len(nodes) - 1):
        start = nodes[i]
        end = nodes[i + 1]
        
        # Check if segment is air route
        if '✈️' in end:
            segment_type = 'air'
        else:
            segment_type = 'road'
            
        segments.append({
            'start': start,
            'end': end,
            'type': segment_type
        })
    
    return segments

def get_flight_type(origin, destination):
    """Get flight type (PRIME/GCR) between airports"""
    origin = origin.replace('✈️', '').strip()
    destination = destination.replace('✈️', '').strip()
    
    gcr = flight_connections_df[
        (flight_connections_df['Flight Type'] == 'GCR') &
        (flight_connections_df['Origin City'] == origin) &
        (flight_connections_df['Destination City'] == destination)
    ].any().any()
    
    prime = flight_connections_df[
        (flight_connections_df['Flight Type'] == 'PRIME') &
        (flight_connections_df['Origin City'] == origin) &
        (flight_connections_df['Destination City'] == destination)
    ].any().any()
    
    return {'GCR': gcr, 'PRIME': prime}

def create_map(center_lat=20.5937, center_lon=78.9629):
    """Create base map with India center"""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='cartodbpositron'
    )
    return m

def add_legend(m):
    """Add custom legend to map"""
    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        right: 50px;
        width: 200px;
        height: auto;
        background-color: white;
        border: 2px solid grey;
        z-index: 1000;
        padding: 10px;
        font-size: 14px;
        border-radius: 5px;
        box-shadow: 3px 3px 5px rgba(0,0,0,0.2);
    ">
    <p style="margin-bottom: 5px;"><strong>🗺️ Route Types</strong></p>
    <p style="margin: 2px;">
        <span style="color: red;">●</span> Hub
    </p>
    <p style="margin: 2px;">
        <span style="color: blue;">●</span> Branch
    </p>
    <p style="margin: 2px;">
        <span style="color: green;">●</span> Airport
    </p>
    <p style="margin: 2px;">
        <span style="color: grey;">―――</span> Road Route
    </p>
    <p style="margin: 2px;">
        <span style="color: blue;">- - -</span> PRIME Air
    </p>
    <p style="margin: 2px;">
        <span style="color: green;">- - -</span> GCR Air
    </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

# Main Dashboard
def main():
    # Remove the CSS for background image/logo if present
    # (No background-image CSS will be injected)

    # Add heading and logo at the top
    st.markdown("""
        <style>
        .dashboard-title {
            display: flex;
            align-items: center;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .dashboard-title img {
            margin-left: 18px;
            height: 48px;
        }
        </style>
    """, unsafe_allow_html=True)
    title_col, logo_col = st.columns([8, 1])
    with title_col:
        st.markdown('<div class="dashboard-title">✈️ Air Route Planning Dashboard</div>', unsafe_allow_html=True)
    with logo_col:
        logo_path2 = os.path.join(os.path.dirname(__file__), 'Screenshot 2025-06-30 at 8.24.08 AM.png')
        if os.path.exists(logo_path2):
            st.image(logo_path2, width=90)

    # Add global CSS for moderate font sizes
    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-size: 1.1em !important;
        }
        .big-metric .stMetricValue {
            font-size: 1.6rem !important;
        }
        .table-heading {
            font-weight: bold;
            font-size: 1.3em;
            text-align: left !important;
            padding-left: 0.5em;
        }
        .table-sub {
            font-size: 1em;
            font-weight: 600;
        }
        .small-note {
            font-size: 0.9em;
            color: #666;
            margin-left: 8px;
        }
        th, td {font-size: 1em !important;}
        </style>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2 = st.tabs(["🧭 Route Explorer", "📈 Flow Analysis"])
    
    # Tab 1: Route Explorer
    with tab1:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Origin and Destination Selection
            origin = st.selectbox(
                "Select Origin Branch",
                options=sorted(shipment_df['Origin Branch'].unique())
            )
            
            destination = st.selectbox(
                "Select Destination Branch",
                options=sorted(shipment_df['Destination Branch'].unique())
            )
            
            if origin and destination:
                route_data = shipment_df[
                    (shipment_df['Origin Branch'] == origin) &
                    (shipment_df['Destination Branch'] == destination)
                ]
                
                if not route_data.empty:
                    st.subheader("Route Details")
                    route = route_data.iloc[0]
                    ep_bp_data = route_data[route_data['Product Type'] == 'EP_BP']
                    es_data = route_data[route_data['Product Type'] == 'ES']
                    col1a, col1b = st.columns(2)
                    epbp_day = round(ep_bp_data['Weight (kg)'].sum(), 2) if not ep_bp_data.empty else 0.00
                    es_day = round(es_data['Weight (kg)'].sum(), 2) if not es_data.empty else 0.00
                    with col1a:
                        st.metric("EP_BP (Premium)", f"{epbp_day:.2f} kg")
                    with col1b:
                        st.metric("ES (Standard)", f"{es_day:.2f} kg")

                    # Only show suggested route and total route time if not a Road recommendation
                    if route['Recommendation'] != 'Road':
                        st.write("**Suggested Route:**")
                        st.write(route['Suggested Route'])
                        indirect_time = route['Indirect Total Time'] if 'Indirect Total Time' in route else '--'
                        if indirect_time != '--' and pd.notnull(indirect_time):
                            try:
                                indirect_time_val = float(indirect_time)
                                indirect_time_str = f"{indirect_time_val:.2f}"
                            except Exception:
                                indirect_time_str = str(indirect_time)
                        else:
                            indirect_time_str = '--'
                        st.write(f"**Total Route Time (Via Air):** {indirect_time_str} hrs (Including Air Travel time, Road time & Processing time at different nodes)")

                    if epbp_day == 0.00 and es_day == 0.00:
                        st.info("No Actual Volume exists between OD pair but path can be used in future")
                    else:
                        if route['Recommendation'] == 'Road':
                            st.warning("Road route is preferable for this OD pair")
                        else:
                            st.write("**Suggested Route:**")
                            st.write(route['Suggested Route'])

                        # Volume Analysis Table (Prime/GCR breakdown)
                        st.subheader("Volume Analysis")
                        epbp_month = round(epbp_day * 25, 2)
                        es_month = round(es_day * 25, 2)
                        # Between Airports: Prime/GCR breakdown
                        calc_fm = pd.read_csv('/Users/devashish/sample/LETS_MAKE_IT_FINAL/Calculated_OD_pair_caterogy.csv')
                        actual_fm = pd.read_csv('Total_actual_volume_flow(Flight_Mode) copy.csv')
                        route_str = route['Suggested Route']
                        air_match = re.search(r'([A-Z\-\s]+) ✈️ ([A-Z\-\s]+)', route_str)
                        airport1, airport2 = None, None
                        calc_day_prime = calc_day_gcr = np.nan
                        actual_day_prime = actual_day_gcr = np.nan
                        calc_month_prime = calc_month_gcr = np.nan
                        actual_month_prime = actual_month_gcr = np.nan
                        if air_match:
                            airport1 = air_match.group(1).strip()
                            airport2 = air_match.group(2).strip()
                            # Calculated
                            calc_row = calc_fm[(calc_fm['Origin_Airport'] == airport1) & (calc_fm['Destination_Airport'] == airport2)]
                            if not calc_row.empty:
                                # Use Aeroplane Category to select PRIME/GCR
                                calc_day_prime = round(calc_row[calc_row['Aeroplane Category']=='PRIME']['Volume(kg)'].sum(), 2)
                                calc_day_gcr = round(calc_row[calc_row['Aeroplane Category']=='GCR']['Volume(kg)'].sum(), 2)
                                calc_month_prime = round(calc_day_prime * 25, 2) if not np.isnan(calc_day_prime) else np.nan
                                calc_month_gcr = round(calc_day_gcr * 25, 2) if not np.isnan(calc_day_gcr) else np.nan
                            # Actual
                            actual_row = actual_fm[(actual_fm['Origin Airport'] == airport1) & (actual_fm['Destination Airport'] == airport2)]
                            if not actual_row.empty:
                                actual_day_prime = round(actual_row[actual_row['Aeroplane Category']=='PRIME']['Volume(kg)'].sum(), 2)
                                actual_day_gcr = round(actual_row[actual_row['Aeroplane Category']=='GCR']['Volume(kg)'].sum(), 2)
                                actual_month_prime = round(actual_day_prime * 25, 2) if not np.isnan(actual_day_prime) else np.nan
                                actual_month_gcr = round(actual_day_gcr * 25, 2) if not np.isnan(actual_day_gcr) else np.nan
                        # Table data with headings and indentation
                        table_data = [
                            [f"<span class='table-heading'>Branch Level</span>", "", ""],
                            ["Day Level", f"{epbp_day:.2f} kg", f"{es_day:.2f} kg"],
                            ["Month Level", f"{epbp_month:.2f} kg", f"{es_month:.2f} kg"],
                            [f"<span class='table-heading'>Between Airports</span>", "Prime", "GCR"],
                            ["Day Level - Calculated", f"{calc_day_prime if not np.isnan(calc_day_prime) else '--'} ", f"{calc_day_gcr if not np.isnan(calc_day_gcr) else '--'} "],
                            ["Day Level - Actual", f"{actual_day_prime if not np.isnan(actual_day_prime) else '--'} ", f"{actual_day_gcr if not np.isnan(actual_day_gcr) else '--'} "],
                            ["Month Level - Calculated", f"{calc_month_prime if not np.isnan(calc_month_prime) else '--'} ", f"{calc_month_gcr if not np.isnan(calc_month_gcr) else '--'} "],
                            ["Month Level - Actual", f"{actual_month_prime if not np.isnan(actual_month_prime) else '--'} ", f"{actual_month_gcr if not np.isnan(actual_month_gcr) else '--'} "]
                        ]
                        st.markdown('<style>th, td {font-size: 0.9 em !important;}</style>', unsafe_allow_html=True)
                        st.write(pd.DataFrame(table_data, columns=["Metric", "Prime", "GCR"]).to_html(escape=False, index=False), unsafe_allow_html=True)
        
        with col2:
            # Map
            if origin and destination and not route_data.empty:
                # Make the map larger and add a border
                m = folium.Map(
                    location=[20.5937, 78.9629],
                    zoom_start=5,
                    tiles='cartodbpositron',
                    width=1100,
                    height=750
                )
                # Add a styled border using a rectangle
                folium.Rectangle(
                    bounds=[[6, 67], [37, 98]],
                    color='#d72660',
                    fill=False,
                    weight=5,
                    dash_array='10,10',
                    opacity=0.7
                ).add_to(m)
                # Always show all hubs (red marker)
                for _, hub in hub_coords_df.iterrows():
                    try:
                        lat, lon = float(hub['Latitude']), float(hub['Longitude'])
                        folium.Marker(
                            [lat, lon],
                            popup=hub['Hub Name'],
                            icon=Icon(color='red', icon='info-sign')
                        ).add_to(m)
                    except Exception:
                        continue
                # Parse route into nodes
                route_str = route_data.iloc[0]['Suggested Route']
                nodes = [n.strip() for n in re.split(r'→|✈️', route_str)]
                # Identify node types and get coordinates
                node_coords = []
                for n in nodes:
                    # Try branch
                    match = branch_coords_df[branch_coords_df[branch_coords_df.columns[1]].astype(str).str.upper() == n.upper()]
                    if not match.empty:
                        lat, lon = match.iloc[0]['Latitude'], match.iloc[0]['Longitude']
                        node_coords.append((n, (lat, lon), 'branch'))
                        continue
                    # Try hub
                    match = hub_coords_df[hub_coords_df[hub_coords_df.columns[1]].astype(str).str.upper() == n.upper()]
                    if not match.empty:
                        lat, lon = match.iloc[0]['Latitude'], match.iloc[0]['Longitude']
                        node_coords.append((n, (lat, lon), 'hub'))
                        continue
                    # Try airport
                    match = airport_coords_df[airport_coords_df[airport_coords_df.columns[1]].astype(str).str.upper() == n.upper()]
                    if not match.empty:
                        lat, lon = match.iloc[0]['Latitude'], match.iloc[0]['Longitude']
                        node_coords.append((n, (lat, lon), 'airport'))
                        continue
                # Draw path segments
                segs = re.split(r'(→|✈️)', route_str)
                seg_types = []
                for i in range(0, len(segs)-2, 2):
                    seg_types.append(segs[i+1])
                for i in range(len(node_coords)-1):
                    n1, coord1, t1 = node_coords[i]
                    n2, coord2, t2 = node_coords[i+1]
                    # Fix: check for NaN in both lat/lon
                    if pd.isna(coord1[0]) or pd.isna(coord1[1]) or pd.isna(coord2[0]) or pd.isna(coord2[1]):
                        continue
                    # Add markers
                    if t1 == 'branch':
                        folium.Marker(coord1, popup=n1, icon=Icon(color='blue', icon='info-sign')).add_to(m)
                    elif t1 == 'airport':
                        folium.Marker(coord1, popup=n1, icon=Icon(color='green', icon='info-sign')).add_to(m)
                    # Draw segment
                    seg_type = seg_types[i] if i < len(seg_types) else '→'
                    if seg_type == '→':
                        PolyLine([coord1, coord2], color='grey', weight=3).add_to(m)
                    elif seg_type == '✈️':
                        # Check flight type
                        flight_types = get_flight_type(n1, n2)
                        if flight_types['PRIME']:
                            PolyLine([coord1, coord2], color='blue', weight=3, dash_array='10').add_to(m)
                        if flight_types['GCR']:
                            PolyLine([coord1, coord2], color='green', weight=3, dash_array='10').add_to(m)
                # Add last node marker
                if node_coords:
                    n_last, coord_last, t_last = node_coords[-1]
                    if t_last == 'branch':
                        folium.Marker(coord_last, popup=n_last, icon=Icon(color='blue', icon='info-sign')).add_to(m)
                    elif t_last == 'airport':
                        folium.Marker(coord_last, popup=n_last, icon=Icon(color='green', icon='info-sign')).add_to(m)
                add_legend(m)
                st_folium(m, height=750, width=1100)
    
    # Tab 2: Flow Analysis
    with tab2:
        st.subheader("Network Overview")
        col2a, col2b, col2c = st.columns(3)
        # Use actual volume data for metrics
        actual_df = pd.read_csv('Airport_to_airport_volume(Actual).csv')
        total_connections = len(actual_df)
        total_volume = actual_df['Actual Volume'].sum()
        avg_volume = total_volume / total_connections if total_connections else 0
        with col2a:
            st.metric("Total Airport Connections", f"{total_connections:,}")
        with col2b:
            st.metric("Total Volume Flow", f"{total_volume:,.2f} kg")
        with col2c:
            st.metric("Average Volume per Connection", f"{avg_volume:,.2f} kg")
        
        # Volume Comparison Plots
        st.subheader("Volume Analysis")
        
        # Remove the first plot (Updated vs Original Volume by Route)
        # Only keep the Actual vs Calculated Volume Comparison plot
        common_routes = pd.merge(
            actual_flow_df,
            calc_volume_df,
            left_on=['origin_city', 'destination_city'],
            right_on=['Origin_Airport', 'Destination_Airport'],
            how='inner'
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=common_routes['origin_city'] + ' → ' + common_routes['destination_city'],
            y=common_routes['Actual Volume'],
            name='Actual Volume',
            mode='lines+markers'
        ))
        fig2.add_trace(go.Scatter(
            x=common_routes['origin_city'] + ' → ' + common_routes['destination_city'],
            y=common_routes['Volume_kg'],
            name='Calculated Volume',
            mode='lines+markers'
        ))
        fig2.update_layout(
            title='Actual vs Calculated Volume Comparison',
            xaxis_title='Route',
            yaxis_title='Volume (kg)',
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
        # Add the two screenshots below the plot
        st.markdown("#### Additional Insights")
        st.image(["Screenshot 2025-07-01 at 1.25.37 PM.png", "Screenshot 2025-07-01 at 1.25.59 PM.png"], width=900)

if __name__ == "__main__":
    main()
