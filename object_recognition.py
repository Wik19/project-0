import cv2
from ultralytics import YOLO
import socket
import struct
import pickle
import numpy as np

# --- Configuration ---
# I have updated this with your correct Raspberry Pi IP address.
PI_HOST = '192.168.1.42'
PI_PORT = 8485
MODEL_NAME = "yolov8s.pt"
WINDOW_NAME = "Low-Latency Object Recognition"

def main():
    """
    Connects to the Raspberry Pi, receives the video stream,
    and performs real-time object detection.
    """
    print(f"Loading YOLOv8 model: {MODEL_NAME}...")
    try:
        model = YOLO(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return

    # Set up the client socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[*] Connecting to Raspberry Pi at {PI_HOST}:{PI_PORT}...")
    try:
        client_socket.connect((PI_HOST, PI_PORT))
        print("[+] Connection successful.")
    except ConnectionRefusedError:
        print(f"[!] Connection refused. Is the stream_sender.py script running on the Pi?")
        return
    
    data = b""
    payload_size = struct.calcsize(">L")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        # Retrieve the message size
        while len(data) < payload_size:
            packet = client_socket.recv(4096)
            if not packet: break
            data += packet
        
        if not data: break

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]

        # Retrieve the actual frame data
        while len(data) < msg_size:
            data += client_socket.recv(4096)
        
        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Decode the frame
        frame_encoded = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame_encoded, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # Run YOLOv8 inference
        results = model(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow(WINDOW_NAME, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Closing socket and destroying windows.")
    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()