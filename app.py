import cv2
import numpy as np
import streamlit as st
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Streamlit App Title
st.title("Virtual Painter")
st.sidebar.header("Marker and Brush Settings")

# Sidebar controls
upper_hue = st.sidebar.slider("Upper Hue", 0, 180, 153)
upper_saturation = st.sidebar.slider("Upper Saturation", 0, 255, 255)
upper_value = st.sidebar.slider("Upper Value", 0, 255, 255)
lower_hue = st.sidebar.slider("Lower Hue", 0, 180, 64)
lower_saturation = st.sidebar.slider("Lower Saturation", 0, 255, 72)
lower_value = st.sidebar.slider("Lower Value", 0, 255, 49)

colors = [
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
    (0, 255, 255),  # Yellow
]
color_names = ["Blue", "Green", "Red", "Yellow"]

# Update the selected color and sync with session state
selected_color = st.sidebar.radio("Brush Color", color_names, index=0)
if "colorIndex" not in st.session_state:
    st.session_state.colorIndex = color_names.index(selected_color)
else:
    st.session_state.colorIndex = color_names.index(selected_color)

# Initialize persistent states
if "paintWindow" not in st.session_state:
    st.session_state.paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255

if "points" not in st.session_state:
    st.session_state.points = [deque(maxlen=1024) for _ in range(4)]

# Clear painting action
if st.sidebar.button("Clear Painting"):
    st.session_state.paintWindow[:, :] = 255  # Reset the paint window
    st.session_state.points = [deque(maxlen=1024) for _ in range(4)]  # Clear all points

paintWindow = st.session_state.paintWindow
points = st.session_state.points

# Video Processor for Streamlit WebRTC
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Update HSV bounds dynamically
        lower_hsv = np.array([lower_hue, lower_saturation, lower_value])
        upper_hsv = np.array([upper_hue, upper_saturation, upper_value])

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Flip horizontally
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create mask for marker detection
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Detect contours
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(cnts) > 0:
            cnt = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Toolbar actions
            if center and center[1] <= 65:
                if 40 <= center[0] <= 140:  # Clear button (toolbar)
                    st.session_state.points = [deque(maxlen=1024) for _ in range(4)]
                    st.session_state.paintWindow[:, :] = 255
            else:
                if center:
                    points[st.session_state.colorIndex].appendleft(center)

        # Draw on the paint window
        for i, point_set in enumerate(points):
            for j in range(1, len(point_set)):
                if point_set[j - 1] is not None and point_set[j] is not None:
                    cv2.line(img, point_set[j - 1], point_set[j], colors[i], 2)
                    cv2.line(paintWindow, point_set[j - 1], point_set[j], colors[i], 2)

        # Overlay paintWindow on the video feed
        combined = cv2.addWeighted(img, 0.7, paintWindow, 0.3, 0)
        return av.VideoFrame.from_ndarray(combined, format="bgr24")

# WebRTC Configuration for STUN servers
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]}
    ]
}

# Streamlit WebRTC component
webrtc_streamer(key="virtual-painter", video_processor_factory=VideoProcessor, rtc_configuration=RTC_CONFIGURATION)

# Display Paint Window
st.subheader("Paint Window")
st.image(cv2.cvtColor(paintWindow, cv2.COLOR_BGR2RGB), use_column_width=True)

# Fallback to OpenCV for local testing
use_opencv = st.sidebar.checkbox("Use Local Camera (OpenCV)", value=False)
if use_opencv:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access camera.")
            break
        frame = cv2.flip(frame, 1)
        stframe.image(frame, channels="BGR", use_container_width=True)
    cap.release()
