import cv2
from ultralytics import YOLO
import socket
import struct
import pickle
import numpy as np
import serial
import time

# --- Network and Model Configuration ---
PI_HOST = '192.168.1.42'
PI_PORT = 8485
MODEL_NAME = "yolov8s.pt"
WINDOW_NAME = "Real-time Object Recognition"

# --- UART and Servo Configuration ---
# Ensure these match your setup
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 115200
SERVO_PAN_ID = 1  # Servo ID for pan (horizontal)
SERVO_TILT_ID = 0 # Servo ID for tilt (vertical)

# --- Servo Angle Limits ---
# This range (0-180) matches the new NUCLEO firmware
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180

# --- Object Tracking Configuration ---
TARGET_CLASS = 'person' 
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

def initialize_uart():
    """Initializes and returns the serial connection to the Nucleo board."""
    try:
        print(f"Connecting to Nucleo on {SERIAL_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for the connection to establish
        print("Successfully connected to Nucleo.")
        return ser
    except serial.SerialException as e:
        print(f"Error connecting to Nucleo: {e}")
        print("Tracking functionality will be disabled.")
        return None

def main():
    serial_connection = initialize_uart()
    
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
        if serial_connection:
            serial_connection.close()
        return
    
    data = b""
    payload_size = struct.calcsize(">L")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Start with servos centered
    current_pan_angle = 90.0
    current_tilt_angle = 90.0

    while True:
        try:
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

            results = model(frame, conf=0.5, verbose=False)
            
            if serial_connection:
                largest_target = None
                max_area = 0

                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]

                        if class_name == TARGET_CLASS:
                            x1, y1, x2, y2 = box.xyxy[0]
                            area = (x2 - x1) * (y2 - y1)
                            if area > max_area:
                                max_area = area
                                largest_target = (x1, y1, x2, y2)

                if largest_target:
                    x1, y1, x2, y2 = largest_target
                    obj_cx = (x1 + x2) / 2
                    obj_cy = (y1 + y2) / 2
                    
                    frame_cx = FRAME_WIDTH / 2
                    frame_cy = FRAME_HEIGHT / 2

                    error_pan = obj_cx - frame_cx
                    error_tilt = obj_cy - frame_cy

                    # IMPORTANT TUNING PARAMETER:
                    # Your Python output shows the angle immediately jumping to 180.
                    # This means your gain is too high for the large frame resolution.
                    # Let's start with a much smaller value.
                    P_GAIN = 0.005 # <-- START WITH A VERY SMALL VALUE and increase if too slow
                    current_pan_angle -= error_pan * P_GAIN
                    current_tilt_angle += error_tilt * P_GAIN

                    # --- Clamp Angles to Valid Range ---
                    current_pan_angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, current_pan_angle))
                    current_tilt_angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, current_tilt_angle))

                    # =======================================================================
                    # <-- THIS IS THE CORRECT LOGIC FOR YOUR NEW NUCLEO FIRMWARE
                    # =======================================================================
                    pan_cmd = f"s{SERVO_PAN_ID}{int(current_pan_angle)}\n"
                    tilt_cmd = f"s{SERVO_TILT_ID}{int(current_tilt_angle)}\n"
                    
                    serial_connection.write(pan_cmd.encode())
                    serial_connection.write(tilt_cmd.encode())
                    print(f"Target at ({obj_cx:.0f}, {obj_cy:.0f}). Sent commands: {pan_cmd.strip()}, {tilt_cmd.strip()}")

            annotated_frame = results[0].plot()
            cv2.imshow(WINDOW_NAME, annotated_frame)
            
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
    if serial_connection:
        center_cmd_pan = f"s{SERVO_PAN_ID}90\n"
        center_cmd_tilt = f"s{SERVO_TILT_ID}90\n"
        serial_connection.write(center_cmd_pan.encode())
        serial_connection.write(center_cmd_tilt.encode())
        time.sleep(0.1)
        serial_connection.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()