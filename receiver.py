import cv2
import time

# raspi command to start stream
# rpicam-vid -t 0 --inline -o - | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.11 port=5000

def main():
    # GStreamer pipeline, same as before
    gst_pipeline = (
        "udpsrc port=5000 ! "
        "application/x-rtp,media=video,payload=96,encoding-name=H264,clock-rate=90000 ! "
        "rtph264depay ! "
        "h264parse ! "
        "avdec_h264 ! "
        "videoconvert ! "
        "appsink drop=true"
    )

    print("--- Attempting to open GStreamer pipeline ---")
    print(f"Pipeline: {gst_pipeline}")
    
    # Create a VideoCapture object
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("\n--- ERROR: Failed to open pipeline. ---")
        print("Check the following:")
        print("1. Is the GStreamer command running on the Raspberry Pi?")
        print("2. Is the IP address in the Pi's command correct for this PC?")
        print("3. Is a firewall blocking UDP port 5000?")
        return

    print("\n--- Pipeline opened successfully. Starting stream display ---")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("--- Waiting for frame or stream ended... ---")
            # Brief pause to prevent a tight loop from consuming CPU
            time.sleep(0.1) 
            continue

        # Display the resulting frame in a window
        cv2.imshow('Raspberry Pi Stream', frame)

        # Press 'q' on the keyboard to exit the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    print("--- Cleaning up and closing windows ---")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()