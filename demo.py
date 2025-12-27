import streamlit as st
import folium
from streamlit_folium import st_folium, folium_static  # Use st_folium for interactive draw
import pandas as pd
import joblib
from query_calculate import exact_count
from query_generate import generate_random_query
from estimation import optimized_laqp_estimate
import folium.plugins as plugins

model_data = joblib.load("12_27_uber.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
best_alpha = model_data["alpha"]
training_data_query_log = model_data["training_query"]
sample = model_data["sample"]

file_path = './data/all_uber.csv'  # Download from URL above
data = pd.read_csv(file_path)
# Change to numerical timestamp
data['datetime'] = pd.to_datetime(data['Date/Time'])
min_dt = data['datetime'].min()
max_dt = data['datetime'].max()
data['timestamp'] = (data['datetime'] - min_dt).dt.total_seconds()
data = data[['timestamp', 'Lat', 'Lon', 'Base']].dropna()  # Ignore Base for now
full_data_size = len(data)
print(f"Dataset loaded: {full_data_size} rows")

# Dimensions for ranges (3D: time + spatial bbox)
dimensions = ['timestamp', 'Lat', 'Lon']

print(data.head())
# Load data, model, etc. (from above code; assume globals)

st.title("LAQP Uber Rides COUNT Demo")

# Predefined city regions (with Garden City added)
city_regions = {
    "Full NYC": {
        "min_lat": data['Lat'].min(), "max_lat": data['Lat'].max(),
        "min_lon": data['Lon'].min(), "max_lon": data['Lon'].max()
    },
    "Garden City": {
        "min_lat": 40.71, "max_lat": 40.74,
        "min_lon": -73.65, "max_lon": -73.62
    },
    "Manhattan": {
        "min_lat": 40.70, "max_lat": 40.82,
        "min_lon": -74.02, "max_lon": -73.93
    },
    "Brooklyn": {
        "min_lat": 40.57, "max_lat": 40.73,
        "min_lon": -74.04, "max_lon": -73.83
    },
    "Queens": {
        "min_lat": 40.54, "max_lat": 40.80,
        "min_lon": -73.90, "max_lon": -73.70
    },
    "Bronx": {
        "min_lat": 40.79, "max_lat": 40.92,
        "min_lon": -73.93, "max_lon": -73.77
    },
}

# Select city region
selected_city = st.selectbox("Select City Region", list(city_regions.keys()))

# Initialize session state with full ranges
if 'min_time' not in st.session_state:
    st.session_state.min_time = data['timestamp'].min()
    st.session_state.max_time = data['timestamp'].max()
    st.session_state.min_lat = data['Lat'].min()
    st.session_state.max_lat = data['Lat'].max()
    st.session_state.min_lon = data['Lon'].min()
    st.session_state.max_lon = data['Lon'].max()
    st.session_state.draw_processed = False
    st.session_state.query_run = False
    st.session_state.exact_val = 0
    st.session_state.approx = 0
    st.session_state.diff = 0
    st.session_state.rel_err = 0

# Update sliders based on selected city (but allow manual adjustment)
city_bbox = city_regions[selected_city]
if selected_city != st.session_state.get('last_city', None):
    st.session_state.last_city = selected_city
    city_bbox = city_regions[selected_city]
    
    # 更新範圍
    st.session_state.min_lat = city_bbox["min_lat"]
    st.session_state.max_lat = city_bbox["max_lat"]
    st.session_state.min_lon = city_bbox["min_lon"]
    st.session_state.max_lon = city_bbox["max_lon"]
    
    # === 關鍵：強制清除舊的 query 結果和 points ===
    st.session_state.query_run = False
    st.session_state.exact_val = 0
    st.session_state.approx = 0
    st.session_state.diff = 0
    st.session_state.rel_err = 0
    st.session_state.last_successful_query = None  # 也可一併清除
    
    st.rerun()

# Calendar for time range
min_datetime = min_dt.date()
max_datetime = max_dt.date()
start_date = st.date_input("Start Date", value=min_datetime, min_value=min_datetime, max_value=max_datetime)
end_date = st.date_input("End Date", value=max_datetime, min_value=min_datetime, max_value=max_datetime)

# Convert to timestamps
start_datetime = pd.to_datetime(start_date)
end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
min_time = (start_datetime - min_dt).total_seconds()
max_time = (end_datetime - min_dt).total_seconds()

if min_time > max_time:
    st.error("Start date must be before end date.")
else:
    st.session_state.min_time = min_time
    st.session_state.max_time = max_time

min_lat, max_lat = st.slider(
    "Lat Range (BBox)", 
    data['Lat'].min(), data['Lat'].max(), 
    (st.session_state.min_lat, st.session_state.max_lat)
)
min_lon, max_lon = st.slider(
    "Lon Range (BBox)", 
    data['Lon'].min(), data['Lon'].max(), 
    (st.session_state.min_lon, st.session_state.max_lon)
)

# Update session state
st.session_state.min_time = min_time
st.session_state.max_time = max_time
st.session_state.min_lat = min_lat
st.session_state.max_lat = max_lat
st.session_state.min_lon = min_lon
st.session_state.max_lon = max_lon

# Random Query Button
if st.button("Generate Random Query"):
    random_query = generate_random_query(data, dimensions)
    st.session_state.min_time = random_query['timestamp'][0]
    st.session_state.max_time = random_query['timestamp'][1]
    st.session_state.min_lat = random_query['Lat'][0]
    st.session_state.max_lat = random_query['Lat'][1]
    st.session_state.min_lon = random_query['Lon'][0]
    st.session_state.max_lon = random_query['Lon'][1]
    st.session_state.query_run = False  # Reset query flag on random
    st.rerun()  # Rerun to update sliders and map

# Interactive Map for Drawing BBox
st.subheader("Draw Custom BBox on Map (Optional)")
draw_map = folium.Map(location=[(data['Lat'].min()+data['Lat'].max())/2, (data['Lon'].min()+data['Lon'].max())/2], zoom_start=8)
plugins.Draw(
    export=False,
    draw_options={'rectangle': True, 'circle': False, 'circlemarker': False, 'marker': False, 'polygon': False, 'polyline': False}
).add_to(draw_map)

# Render interactive map and get output
map_data = st_folium(draw_map, key="draw_map", width=700, height=500)
# st.write("Debug: Map Data Output", map_data)

# If user drew a rectangle, and it's new (not processed)
if map_data and map_data.get('last_active_drawing') and map_data['last_active_drawing']['geometry']['type'] == 'Polygon' and not st.session_state.draw_processed:
    coords = map_data['last_active_drawing']['geometry']['coordinates'][0]
    drawn_min_lon = min(c[0] for c in coords)
    drawn_max_lon = max(c[0] for c in coords)
    drawn_min_lat = min(c[1] for c in coords)
    drawn_max_lat = max(c[1] for c in coords)
    
    # Check if bounds differ from current
    if (drawn_min_lat != st.session_state.min_lat or drawn_max_lat != st.session_state.max_lat or
        drawn_min_lon != st.session_state.min_lon or drawn_max_lon != st.session_state.max_lon):
        # Update session state with drawn bounds
        st.session_state.min_lat = drawn_min_lat
        st.session_state.max_lat = drawn_max_lat
        st.session_state.min_lon = drawn_min_lon
        st.session_state.max_lon = drawn_max_lon
        st.session_state.draw_processed = True  # Mark as processed
        st.rerun()  # Rerun to refresh sliders and map
    else:
        st.session_state.draw_processed = True  # Mark even if same to prevent loop

else:
    st.session_state.draw_processed = False  # Reset for next draw

# Map for BBox Preview (integrate matching points if query run)
st.subheader("Query BBox Preview (Red Box Shows Current Range; Points Shown After Query)")
m = folium.Map(location=[(data['Lat'].min()+data['Lat'].max())/2, (data['Lon'].min()+data['Lon'].max())/2], zoom_start=8)  # Lower zoom to show full box
# Dynamic red bbox based on sliders
folium.Rectangle(bounds=[[st.session_state.min_lat, st.session_state.min_lon], [st.session_state.max_lat, st.session_state.max_lon]], color="red", fill=False).add_to(m)
m.fit_bounds([[st.session_state.min_lat, st.session_state.min_lon], [st.session_state.max_lat, st.session_state.max_lon]])

# If query has been run, add matching points
if st.session_state.query_run:
    matching = data[(data['timestamp'] >= st.session_state.min_time) & (data['timestamp'] <= st.session_state.max_time) &
                    (data['Lat'] >= st.session_state.min_lat) & (data['Lat'] <= st.session_state.max_lat) &
                    (data['Lon'] >= st.session_state.min_lon) & (data['Lon'] <= st.session_state.max_lon)]
    subsample = matching.sample(min(1000, len(matching)))  # Limit to 1000 for performance
    for _, row in subsample.iterrows():
        folium.Marker([row['Lat'], row['Lon']]).add_to(m)

folium_static(m)

query = {'timestamp': (st.session_state.min_time, st.session_state.max_time), 'Lat': (st.session_state.min_lat, st.session_state.max_lat), 'Lon': (st.session_state.min_lon, st.session_state.max_lon)}
# === 新增這段：自動偵測 query 改變就清除舊 points ===
if 'last_successful_query' not in st.session_state:
    st.session_state.last_successful_query = None

if (st.session_state.query_run and 
    st.session_state.last_successful_query != query):
    st.session_state.query_run = False
    st.session_state.exact_val = 0
    st.session_state.approx = 0
    st.session_state.diff = 0
    st.session_state.rel_err = 0

if st.button("Query Search"):
    approx, entry = optimized_laqp_estimate(training_data_query_log, query, sample, dimensions, model, scaler, full_data_size, best_alpha)
    exact_val = exact_count(query, data)
    diff = abs(approx - exact_val)
    rel_err = diff / exact_val if exact_val > 0 else 0
    
    st.session_state.exact_val = exact_val
    st.session_state.approx = approx
    st.session_state.diff = diff
    st.session_state.rel_err = rel_err
    st.session_state.query_run = True
    st.session_state.last_successful_query = query  # 記錄本次成功的 query
    st.rerun()  # Rerun to update preview map with points

# Display results if query run
if st.session_state.query_run:
    st.write(f"Exact COUNT: {st.session_state.exact_val}")
    st.write(f"LAQP Approx COUNT: {st.session_state.approx:.0f}")
    st.write(f"Difference: {st.session_state.diff:.4f} (Rel Error: {st.session_state.rel_err:.4f})")