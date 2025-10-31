import socket
import struct
import pickle
import cv2
from picamera2 import Picamera2 # Import the new library
import time

# --- Configuration ---
HOST = '0.0.0.0'
PORT = 8485
RESOLUTION = (1920, 1080)
JPEG_QUALITY = 90

def main():
    """
    Captures video using Picamera2 and streams it over a TCP socket.
    """
    # --- Picamera2 Setup ---
    picam2 = Picamera2()
    # Create a configuration for video recording
    config = picam2.create_video_configuration(main={"size": RESOLUTION})
    picam2.configure(config)
    
    print("Starting camera...")
    picam2.start()
    # The camera needs a moment to adjust to light levels
    time.sleep(1)
    print("Camera started successfully.")

    # --- Socket Setup ---
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"[*] Listening for a client on {HOST}:{PORT}")

    conn, addr = server_socket.accept()
    print(f"[+] Accepted connection from {addr}")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    try:
        while True:
            # Capture a frame as a NumPy array (OpenCV compatible)
            frame = picam2.capture_array()
            
            # Convert from RGB (picamera2) to BGR (OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Encode the frame as JPEG
            result, frame_encoded = cv2.imencode('.jpg', frame_bgr, encode_param)
            data = pickle.dumps(frame_encoded, 0)
            size = len(data)

            # Send the data
            conn.sendall(struct.pack(">L", size) + data)

    except (BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
        print("Client disconnected or script stopped.")
    finally:
        print("Closing connection, stopping camera.")
        picam2.stop()
        conn.close()
        server_socket.close()

if __name__ == '__main__':
    main()