import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Streamlit App Title
st.title("Virtual Painter with Camera Input")
st.sidebar.header("Marker and Brush Settings")

# Sidebar controls for HSV color range
upper_hue = st.sidebar.slider("Upper Hue", 0, 180, 153)
upper_saturation = st.sidebar.slider("Upper Saturation", 0, 255, 255)
upper_value = st.sidebar.slider("Upper Value", 0, 255, 255)
lower_hue = st.sidebar.slider("Lower Hue", 0, 180, 64)
lower_saturation = st.sidebar.slider("Lower Saturation", 0, 255, 72)
lower_value = st.sidebar.slider("Lower Value", 0, 255, 49)

# Brush color selection
colors = [
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
    (0, 255, 255),  # Yellow
]
color_names = ["Blue", "Green", "Red", "Yellow"]
selected_color = st.sidebar.radio("Brush Color", color_names, index=0)
color_index = color_names.index(selected_color)

# Persistent storage for points and paint window
if "paintWindow" not in st.session_state:
    st.session_state.paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255
if "points" not in st.session_state:
    st.session_state.points = [deque(maxlen=1024) for _ in range(4)]

# Clear button action
if st.sidebar.button("Clear Painting"):
    st.session_state.paintWindow[:, :] = 255
    st.session_state.points = [deque(maxlen=1024) for _ in range(4)]

paintWindow = st.session_state.paintWindow
points = st.session_state.points

# Camera input widget
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer:
    # Read the image from the input buffer
    img = Image.open(img_file_buffer)
    img = np.array(img)  # Convert to numpy array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create mask for color detection
    lower_hsv = np.array([lower_hue, lower_saturation, lower_value])
    upper_hsv = np.array([upper_hue, upper_saturation, upper_value])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Detect contours
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Add the detected point to the corresponding deque
        if center:
            points[color_index].appendleft(center)

    # Draw points on the paint window
    for i, point_set in enumerate(points):
        for j in range(1, len(point_set)):
            if point_set[j - 1] is not None and point_set[j] is not None:
                cv2.line(img, point_set[j - 1], point_set[j], colors[i], 2)
                cv2.line(paintWindow, point_set[j - 1], point_set[j], colors[i], 2)

    # Combine paintWindow with the image
    combined = cv2.addWeighted(img, 0.7, paintWindow, 0.3, 0)
    st.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

# Display Paint Window
st.subheader("Paint Window")
st.image(cv2.cvtColor(paintWindow, cv2.COLOR_BGR2RGB), use_column_width=True)
