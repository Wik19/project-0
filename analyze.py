import cv2
from ultralytics import YOLO

# FUCKING WORKS!!!!!!!!
# rpicam-vid --width 1920 --height 1080 -t 0 --inline -o - | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.11 port=5000

# --- GStreamer Pipeline with NVIDIA Hardware-Acceleration ---
# This is the pipeline you tested with gst-launch-1.0
pipeline = (
    "udpsrc port=5000 "
    "! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 "
    "! rtph264depay "
    "! h264parse "
    "! nvh264dec "  # Using your NVIDIA decoder
    "! videoconvert "
    "! video/x-raw, format=BGR "
    "! appsink"
)

# --- YOLO Model ---
print("Loading YOLO model...")
model = YOLO('yolov8x.pt') 
print("Model loaded.")

# --- Video Capture ---
# Make sure your Pi is streaming BEFORE you run this!
print("Connecting to video stream...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    print("Check GStreamer pipeline syntax and that the Pi is streaming.")
    exit()

print("Successfully opened video stream! Starting analysis...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream end or error. Exiting.")
        break

    # --- Your YOLO Analysis ---
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