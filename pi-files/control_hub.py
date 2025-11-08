import socket
import struct
import pickle
import cv2
from picamera2 import Picamera2
import time
import threading
import serial

# --- Configuration ---
VIDEO_HOST = '0.0.0.0'  # Listen on all available network interfaces
VIDEO_PORT = 8485
COMMAND_HOST = '0.0.0.0'
COMMAND_PORT = 8486 # A separate port for receiving commands

RESOLUTION = (1920, 1080)
JPEG_QUALITY = 90

# --- NUCLEO UART Configuration ---
# Update this to your Pi's serial port name (e.g., '/dev/ttyACM0' or '/dev/ttyS0')
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

def initialize_serial():
    """Tries to connect to the NUCLEO board and returns the serial object."""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[+] Successfully connected to NUCLEO on {SERIAL_PORT}")
        return ser
    except serial.SerialException as e:
        print(f"[!] FAILED to connect to NUCLEO: {e}")
        print("[!] Commands will be received but not forwarded.")
        return None

def video_stream_thread():
    """
    Thread function dedicated to capturing and streaming video to the PC.
    This is your original script, slightly modified to run in a thread.
    """
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": RESOLUTION})
    picam2.configure(config)
    
    print("[Video Thread] Starting camera...")
    picam2.start()
    time.sleep(1)
    print("[Video Thread] Camera started successfully.")

    video_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    video_server_socket.bind((VIDEO_HOST, VIDEO_PORT))
    video_server_socket.listen(1)
    print(f"[*] [Video Thread] Listening for a video client on {VIDEO_HOST}:{VIDEO_PORT}")

    conn, addr = video_server_socket.accept()
    print(f"[+] [Video Thread] Accepted video connection from {addr}")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            result, frame_encoded = cv2.imencode('.jpg', frame_bgr, encode_param)
            data = pickle.dumps(frame_encoded, 0)
            size = len(data)
            conn.sendall(struct.pack(">L", size) + data)
    except (BrokenPipeError, ConnectionResetError):
        print("[!] [Video Thread] Client disconnected.")
    finally:
        print("[Video Thread] Closing connection and stopping camera.")
        picam2.stop()
        conn.close()
        video_server_socket.close()

def command_listener_thread(serial_connection):
    """
    Thread function dedicated to listening for commands from the PC
    and writing them to the NUCLEO's serial port.
    """
    command_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    command_server_socket.bind((COMMAND_HOST, COMMAND_PORT))
    command_server_socket.listen(1)
    print(f"[*] [Command Thread] Listening for a command client on {COMMAND_HOST}:{COMMAND_PORT}")

    while True:
        try:
            conn, addr = command_server_socket.accept()
            print(f"[+] [Command Thread] Accepted command connection from {addr}")
            
            while True:
                # Receive command data from the PC
                data = conn.recv(1024)
                if not data:
                    print("[!] [Command Thread] Client disconnected.")
                    break # Break inner loop to wait for new connection
                
                # If NUCLEO is connected, forward the command
                if serial_connection:
                    serial_connection.write(data)
                
                # Optional: Print the command for debugging on the Pi
                print(f"[->] Received command: {data.decode('utf-8').strip()}")

        except (BrokenPipeError, ConnectionResetError):
            print("[!] [Command Thread] Client connection lost. Waiting for new connection.")
        except Exception as e:
            print(f"[!] [Command Thread] An error occurred: {e}")
            break
            
    command_server_socket.close()
    if serial_connection:
        serial_connection.close()
    print("[Command Thread] Shutting down.")


if __name__ == '__main__':
    # Initialize the serial connection to the NUCLEO first
    ser = initialize_serial()

    # Create the thread for the video stream
    video_thread = threading.Thread(target=video_stream_thread)
    # Create the thread for the command listener, passing it the serial object
    command_thread = threading.Thread(target=command_listener_thread, args=(ser,))

    # Start both threads
    video_thread.start()
    command_thread.start()

    # Wait for both threads to complete (they won't, but this keeps the main script alive)
    video_thread.join()
    command_thread.join()