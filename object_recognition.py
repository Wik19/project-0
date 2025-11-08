import cv2
from ultralytics import YOLO
import socket
import struct
import pickle
import numpy as np

# --- Network and Model Configuration ---
PI_HOST = '192.168.1.42'
PI_PORT = 8485
MODEL_NAME = "yolov8s.pt"
WINDOW_NAME = "Real-time Object Recognition"

def main():
    # --- Initialize Model and Video Stream ---
    print(f"Loading YOLOv8 model: {MODEL_NAME}...")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    print(f"Connecting to Raspberry Pi at {PI_HOST}:{PI_PORT}...")
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((PI_HOST, PI_PORT))
        print("Connection successful.")
    except Exception as e:
        print(f"Error connecting to Raspberry Pi: {e}")
        return
    
    data = b""
    payload_size = struct.calcsize(">L")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # --- Main Display Loop ---
    while True:
        try:
            # --- Receive Frame from Pi ---
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet: raise ConnectionError("Pi disconnected.")
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += client_socket.recv(4096)
            
            frame_data = data[:msg_size]
            data = data[msg_size:]
            
            frame_encoded = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame = cv2.imdecode(frame_encoded, cv2.IMREAD_COLOR)
            if frame is None:
                print("Received an empty frame.")
                continue

            # --- Object Detection and Visualization ---
            results = model(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()
            
            # Display the frame with detected objects
            cv2.imshow(WINDOW_NAME, annotated_frame)
            
            # Check for 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, exiting.")
                break

        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            print(f"Connection to Raspberry Pi lost: {e}")
            break
        except KeyboardInterrupt:
            print("Program interrupted by user.")
            break

    # --- Cleanup ---
    print("Closing...")
    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()