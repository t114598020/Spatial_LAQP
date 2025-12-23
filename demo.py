import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import joblib
from query_calculate import exact_count, sample_count
import numpy as np

def range_distance(q1, q2):
    vec1 = [q1[dim][i] for dim in dimensions for i in range(2)]
    vec2 = [q2[dim][i] for dim in dimensions for i in range(2)]
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

model_data = joblib.load("12_23_uber.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
best_alpha = model_data["alpha"]
training_data_query_log = model_data["training_query"]
sample = model_data["sample"]

file_path = 'all_uber.csv'  # Download from URL above
data = pd.read_csv(file_path)
# Change to numerical timestamp
data['datetime'] = pd.to_datetime(data['Date/Time'])
min_dt = data['datetime'].min()
data['timestamp'] = (data['datetime'] - min_dt).dt.total_seconds()
data = data[['timestamp', 'Lat', 'Lon', 'Base']].dropna()  # Ignore Base for now
full_data_size = len(data)
print(f"Dataset loaded: {full_data_size} rows")

# Optimized (hybrid with best_alpha)
def laqp_count(query, alpha=best_alpha):
    vec = [query[dim][i] for dim in dimensions for i in range(2)]
    vec = np.array([vec])
    scaled = scaler.transform(vec)  # Assuming your scaler
    pred_error = model.predict(scaled)[0]
    
    best_entry = min(training_data_query_log, key=lambda e: 
        alpha * abs(e['error'] - pred_error) + 
        (1 - alpha) * range_distance(query, e['query']))
    
    sample_new = sample_count(query, sample, full_data_size)
    sample_opt = best_entry['estimate']
    opt_est = best_entry['exact'] + (sample_new - sample_opt)
    
    print(f"Optimized LAQP estimate: {opt_est:.2f}")
    print(f"Chosen historical error: {best_entry['error']:.2f}")
    return opt_est, best_entry

# Dimensions for ranges (3D: time + spatial bbox)
dimensions = ['timestamp', 'Lat', 'Lon']

print(data.head())
# Load data, model, etc. (from above code; assume globals)

st.title("LAQP Uber Rides COUNT Demo")

# Inputs (use st sliders for time/lat/lon)
min_time, max_time = st.slider("Time Range (Unix seconds)", data['timestamp'].min(), data['timestamp'].max(), (data['timestamp'].min(), data['timestamp'].max()))
min_lat, max_lat = st.slider("Lat Range (BBox)", data['Lat'].min(), data['Lat'].max(), (40.0, 41.5))
min_lon, max_lon = st.slider("Lon Range (BBox)", data['Lon'].min(), data['Lon'].max(), (-74.0, -71.5))
# Map for BBox Draw
st.subheader("Query BBox Preview (Adjust Sliders to Resize Red Box)")
m = folium.Map(location=[40.75, -73.95], zoom_start=12)  # NYC center
# Dynamic red bbox based on sliders
folium.Rectangle(bounds=[[min_lat, min_lon], [max_lat, max_lon]], color="red", fill=False).add_to(m)
folium_static(m)
query = {'timestamp': (min_time, max_time), 'Lat': (min_lat, max_lat), 'Lon': (min_lon, max_lon)}

if st.button("Query"):
    approx, entry = laqp_count(query)
    exact_val = exact_count(query, data)
    diff = abs(approx - exact_val)
    rel_err = diff / exact_val if exact_val > 0 else 0
    
    st.write(f"Exact COUNT: {exact_val}")
    st.write(f"LAQP Approx COUNT: {approx}")
    st.write(f"Difference: {diff} (Rel Error: {rel_err:.4f})")
    
    # Visualize matching points on map with red bbox
    st.subheader("Query Matching Points (With Red BBox)")
    matching = data[(data['timestamp'] >= min_time) & (data['timestamp'] <= max_time) &
                    (data['Lat'] >= min_lat) & (data['Lat'] <= max_lat) &
                    (data['Lon'] >= min_lon) & (data['Lon'] <= max_lon)]
    center_lat = (min_lat + max_lat)/2
    center_lon = (min_lon + max_lon)/2
    viz_m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add red bbox to query map
    folium.Rectangle(bounds=[[min_lat, min_lon], [max_lat, max_lon]], color="red", fill=False).add_to(viz_m)
    
    # Add markers (subsample if too many)
    subsample = matching.sample(min(1000, len(matching)))  # Limit to 1000 for performance
    for _, row in subsample.iterrows():
        folium.Marker([row['Lat'], row['Lon']]).add_to(viz_m)
    folium_static(viz_m)