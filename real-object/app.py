import streamlit as st
from PIL import Image
import torch
import torch.serialization
from ultralytics import YOLO
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
import tempfile
import cv2
import numpy as np
import pandas as pd
import altair as alt
import time
import io

# Ensure compatibility with the ultralytics package
try:
    import ultralytics
except ImportError:
    raise ImportError("The 'ultralytics' package is not installed. Please install it using 'pip install ultralytics'.")

# Add necessary classes to safe globals
torch.serialization.add_safe_globals([
    DetectionModel,
    nn.modules.container.Sequential,
    nn.Sequential,
    nn.ModuleList,
    nn.Module,
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.LeakyReLU,
    nn.Upsample
])

# Streamlit config
st.set_page_config(page_title="Crowd Tracker", layout="centered")
st.title("👥 Real-Time Crowd Detection and Counting with YOLOv8")

@st.cache_resource
def load_model(model_path):
    try:
        original_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        torch.load = patched_load
        model = YOLO(model_path)
        torch.load = original_load
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize session state for crowd counts
if 'crowd_counts' not in st.session_state:
    st.session_state['crowd_counts'] = {}

# Initialize session state for total people counted
if 'total_people_counted' not in st.session_state:
    st.session_state['total_people_counted'] = 0

# Initialize session state for time series data
if 'time_series_data' not in st.session_state:
    st.session_state['time_series_data'] = []

# Load model
with st.spinner("Loading model..."):
    try:
        with torch.serialization.safe_globals([
            DetectionModel,
            nn.modules.container.Sequential,
            nn.Sequential,
            nn.ModuleList,
            nn.Module
        ]):
            model = load_model("yolov8n.pt")

        if model is None:
            st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Function to update crowd counts and time series data
def update_crowd_data(results):
    detected_classes = results[0].names
    boxes = results[0].boxes
    current_frame_counts = {}
    if len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls.item())
            class_name = detected_classes[class_id]
            current_frame_counts[class_name] = current_frame_counts.get(class_name, 0) + 1

    timestamp = time.time()
    for person_type, count in current_frame_counts.items():
        st.session_state['crowd_counts'][person_type] = st.session_state['crowd_counts'].get(person_type, 0) + count
        st.session_state['total_people_counted'] += count
        st.session_state['time_series_data'].append({'timestamp': timestamp, 'person_type': person_type, 'count': 1}) # Record each detection

# Input source selection
input_source = st.radio("Select input source:", ("Image", "Video", "Camera"))

# Upload file (image or video)
if input_source in ["Image", "Video", "Camera"]:
    uploaded_file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png", "mp4"])
else:
    uploaded_file = None

# Use phone camera
if input_source == "Camera":
    camera_input = st.camera_input("Capture from Camera")
    if camera_input:
        uploaded_file = camera_input  # Treat camera input as an uploaded file

# Placeholder for real-time chart
realtime_chart_placeholder = st.empty()

# Process image
if uploaded_file and model and input_source == "Image":
    with st.spinner("Processing image..."):
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            results = model(image)
            results_img = results[0].plot()
            st.image(results_img, caption="Detected Image", use_container_width=True)

            st.subheader("Detection Results")
            boxes = results[0].boxes

            if len(boxes) > 0:
                update_crowd_data(results)
                for i, box in enumerate(boxes):
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    class_name = results[0].names[class_id]
                    st.write(f"Object {i+1}: {class_name} - Confidence: {confidence:.2f}")
            else:
                st.info("No objects detected.")

            st.subheader("Crowd Count Summary")
            st.write(f"Total People Counted: {st.session_state['total_people_counted']}")
            if st.session_state['crowd_counts']:
                df_summary = pd.DataFrame(list(st.session_state['crowd_counts'].items()), columns=['Person Type', 'Count'])
                chart_summary = alt.Chart(df_summary).mark_bar().encode(
                    x='Person Type',
                    y='Count',
                    tooltip=['Person Type', 'Count']
                ).properties(
                    title='Total Number of People Counted by Type'
                )
                st.altair_chart(chart_summary, use_container_width=True)
            else:
                st.info("No people detected yet to show summary.")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Process video (including camera input)
elif (uploaded_file and model and input_source in ["Video", "Camera"]):
    with st.spinner("Processing video..."):
        try:
            # Save the uploaded video or camera input to a temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

            cap = cv2.VideoCapture(temp_file_path)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                result_frame = results[0].plot()
                stframe.image(result_frame, channels="RGB", use_container_width=True)

                update_crowd_data(results)

                # Create real-time chart
                if st.session_state['time_series_data']:
                    df_realtime = pd.DataFrame(st.session_state['time_series_data'])
                    # Group by time (e.g., every few seconds) and person type for a smoother chart
                    df_realtime['time_interval'] = df_realtime['timestamp'].astype(int) // 5 * 5 # Group by 5-second intervals
                    df_grouped = df_realtime.groupby(['time_interval', 'person_type']).size().reset_index(name='count')

                    chart_realtime = alt.Chart(df_grouped).mark_line(point=True).encode(
                        x=alt.X('time_interval:T', title='Time'),
                        y=alt.Y('count:Q', title='People Detected'),
                        color='person_type:N',
                        tooltip=['time_interval:T', 'person_type', 'count']
                    ).properties(
                        title='Real-Time Crowd Detection Over Time'
                    )
                    realtime_chart_placeholder.altair_chart(chart_realtime, use_container_width=True)
                else:
                    realtime_chart_placeholder.info("No people detected in the current video stream.")

            cap.release()

            st.subheader("Final Crowd Count Summary")
            st.write(f"Total People Counted: {st.session_state['total_people_counted']}")
            if st.session_state['crowd_counts']:
                df_final_summary = pd.DataFrame(list(st.session_state['crowd_counts'].items()), columns=['Person Type', 'Count'])
                chart_final_summary = alt.Chart(df_final_summary).mark_bar().encode(
                    x='Person Type',
                    y='Count',
                    tooltip=['Person Type', 'Count']
                ).properties(
                    title='Total Number of People Detected in Video'
                )
                st.altair_chart(chart_final_summary, use_container_width=True)
            else:
                st.info("No people detected in the video.")

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

def save_crowd_results_to_csv(time_series_data):
    # Enhanced DataFrame with Frame Number, Person Type, Count, and Timestamp
    df = pd.DataFrame(time_series_data)
    if not df.empty:
        # Assign frame numbers (1-based index for each detection)
        df['Frame Number'] = range(1, len(df) + 1)
        # Convert timestamp to readable time
        df['Time'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
        # Group by frame and person type for summary
        df_grouped = df.groupby(['Frame Number', 'Time', 'person_type']).agg({'count': 'sum'}).reset_index()
        df_grouped.rename(columns={
            'person_type': 'Person Type',
            'count': 'Person Count in Crowd',
            'Time': 'Detection Time'
        }, inplace=True)
        # Reorder columns for clarity
        df_grouped = df_grouped[['Frame Number', 'Detection Time', 'Person Type', 'Person Count in Crowd']]
        # Calculate total count for each person type
        total_counts = df_grouped.groupby('Person Type')['Person Count in Crowd'].sum().reset_index()
        total_counts['Frame Number'] = 'TOTAL'
        total_counts['Detection Time'] = ''
        total_counts = total_counts[['Frame Number', 'Detection Time', 'Person Type', 'Person Count in Crowd']]
        # Append total row(s) to the DataFrame
        df_final = pd.concat([df_grouped, total_counts], ignore_index=True)
        return df_final
    else:
        return pd.DataFrame(columns=['Frame Number', 'Detection Time', 'Person Type', 'Person Count in Crowd'])

# After the video/image processing blocks, add download button
if st.session_state['time_series_data']:
    df_crowd_csv = save_crowd_results_to_csv(st.session_state['time_series_data'])
    st.download_button(
        label="👥 Download Crowd Count Results as CSV",
        data=df_crowd_csv.to_csv(index=False),
        file_name="crowd_count_results.csv",
        mime="text/csv"
    )