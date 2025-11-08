import cv2
from ultralytics import YOLO
import socket
import struct
import pickle
import numpy as np
# NOTE: 'serial' and 'time' are no longer needed for direct control

# --- Network and Model Configuration ---
PI_HOST = '192.168.1.42' # IP address of your Raspberry Pi
PI_VIDEO_PORT = 8485
PI_COMMAND_PORT = 8486 # The new port for sending commands

MODEL_NAME = "yolov8s.pt"
WINDOW_NAME = "Real-time Object Recognition"

# --- Servo Configuration (Must match NUCLEO firmware) ---
SERVO_PAN_ID = 1  # As you discovered, Pan is ID 1
SERVO_TILT_ID = 0 # and Tilt is ID 0

SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180

# --- Object Tracking Configuration ---
TARGET_CLASS = 'person' 
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
P_GAIN = 0.005 # Proportional gain, may need tuning

def send_command_to_pi(command):
    """
    Connects to the Raspberry Pi's command server, sends a single command,
    and closes the connection.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((PI_HOST, PI_COMMAND_PORT))
            s.sendall(command.encode('utf-8'))
    except ConnectionRefusedError:
        print(f"[!] Command failed: Connection to {PI_HOST}:{PI_COMMAND_PORT} refused. Is the control_hub.py script running on the Pi?")
    except Exception as e:
        print(f"[!] An error occurred while sending command: {e}")

def main():
    # --- NO MORE SERIAL INITIALIZATION HERE ---

    # --- Initialize Model ---
    print(f"Loading YOLOv8 model: {MODEL_NAME}...")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # --- Connect to Raspberry Pi Video Stream ---
    print(f"Connecting to Raspberry Pi for video at {PI_HOST}:{PI_VIDEO_PORT}...")
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((PI_HOST, PI_VIDEO_PORT))
        print("Video connection successful.")
    except Exception as e:
        print(f"Error connecting to Raspberry Pi video stream: {e}")
        return
    
    data = b""
    payload_size = struct.calcsize(">L")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    current_pan_angle = 90.0
    current_tilt_angle = 90.0

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

            # --- Object Detection and Tracking Logic (Unchanged) ---
            results = model(frame, conf=0.5, verbose=False)
            
            largest_target = None
            max_area = 0
            # ... (rest of detection logic is identical)
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
                obj_cx = (largest_target[0] + largest_target[2]) / 2
                obj_cy = (largest_target[1] + largest_target[3]) / 2
                frame_cx = FRAME_WIDTH / 2
                frame_cy = FRAME_HEIGHT / 2
                error_pan = obj_cx - frame_cx
                error_tilt = obj_cy - frame_cy
                current_pan_angle -= error_pan * P_GAIN
                current_tilt_angle += error_tilt * P_GAIN
                current_pan_angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, current_pan_angle))
                current_tilt_angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, current_tilt_angle))

                # =======================================================================
                # <-- MAJOR CHANGE: Sending commands over the network instead of serial
                # =======================================================================
                pan_cmd = f"s{SERVO_PAN_ID}{int(current_pan_angle)}\n"
                tilt_cmd = f"s{SERVO_TILT_ID}{int(current_tilt_angle)}\n"
                
                send_command_to_pi(pan_cmd)
                send_command_to_pi(tilt_cmd)
                
                print(f"Target at ({obj_cx:.0f}, {obj_cy:.0f}). Sent network commands: {pan_cmd.strip()}, {tilt_cmd.strip()}")

            # --- Display Frame ---
            annotated_frame = results[0].plot()
            cv2.imshow(WINDOW_NAME, annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            print(f"Connection to Raspberry Pi lost: {e}")
            break
        except KeyboardInterrupt:
            print("Program interrupted by user.")
            break

    print("Closing...")
    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()