import cv2
from ultralytics import YOLO
import os

# --- TCP Stream Connection Details ---
# Replace with your Raspberry Pi's actual IP address
PI_IP_ADDRESS = "192.168.1.42" 
STREAM_URL = f"tcp://{PI_IP_ADDRESS}:5000"

# --- YOLO Model ---
print("Loading YOLO model...")
model = YOLO('yolov8x.pt') 
print("Model loaded.")

# --- Video Capture ---
# Make sure the Pi is running the TCP server command BEFORE you run this!
print(f"Connecting to TCP stream at {STREAM_URL}...")

# Set FFMPEG options: allow TCP and set a longer timeout for the connection
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,tcp|timeout;5000000"

# Use the FFMPEG backend to connect to the TCP stream
cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    print("Check that the Pi is running the tcpserversink command and the IP is correct.")
    exit()

print("Successfully opened video stream! Starting analysis...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or connection lost. Exiting.")
        break

    # --- Your YOLO Analysis (This will use your RTX 3060 Ti) ---
    results = model(frame, stream=True) 

    # Visualize the results
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("YOLO Real-Time Analysis", annotated_frame)

    # 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Cleaning up and closing...")
cap.release()
cv2.destroyAllWindows()