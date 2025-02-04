import socket
import cv2
import numpy as np
import threading
import time
import struct
import os

HOST = '0.0.0.0'
BACK_CAMERA_PORT = 12345
FRONT_CAMERA_PORT = 12346
DISPLAY_WIDTH = 720   # 固定显示窗口宽度 (竖屏)
DISPLAY_HEIGHT = 1280  # 固定显示窗口高度 (竖屏)

camera_windows = {}

def handle_client(conn, addr, camera_type):
    print(f"Connected to {camera_type} Camera by {addr}")
    window_name = f"Camera Stream from {camera_type} Camera - {addr}"
    camera_windows[window_name] = False

    frame_count = 0
    try:
        while True:
            frame_data = b''
            timestamp = 0
            try:
                timestamp_data = conn.recv(8)
                if not timestamp_data:
                    print(f"Client for {camera_type} camera disconnected (timestamp).")
                    return
                timestamp = struct.unpack('!q', timestamp_data)[0]

                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        print(f"Client for {camera_type} camera disconnected (frame).")
                        return
                    frame_data += chunk
                    if len(frame_data) > 4 and frame_data.endswith(b'\xff\xd9'):
                        break
                    elif not chunk:
                        print(f"Client for {camera_type} camera disconnected unexpectedly.")
                        return

                if not frame_data:
                    print(f"No frame data received for {camera_type} camera, connection might be closed.")
                    break

                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    print(f"解码失败 for {camera_type} camera, frame_data size: {len(frame_data)}")
                else:
                    # 调整帧大小为固定尺寸
                    resized_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                    if not camera_windows[window_name]:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        # 禁用自动调整大小，保持固定比例
                        cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
                        camera_windows[window_name] = True

                    # 计算延迟
                    current_time = int(time.time() * 1000)
                    delay = current_time - timestamp

                    # 在图像上添加延迟信息
                    cv2.putText(resized_frame, f"Delay: {delay} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow(window_name, resized_frame)

                    # 清除控制台上一行
                    print("\033[A                             \033[A")
                    print(f"[{camera_type}] Delay: {delay} ms", end='\r')  # 使用 \r 实现重复覆盖

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1

            except Exception as e:
                print(f"Error processing frame for {camera_type} camera: {e}")
                break

    except Exception as e:
        print(f"Connection error with {addr} for {camera_type} camera: {e}")
    finally:
        conn.close()
        print(f"Connection with {addr} closed for {camera_type} camera")
        if window_name in camera_windows and camera_windows[window_name]:
            cv2.destroyWindow(window_name)
            del camera_windows[window_name]

def start_camera_server(port, camera_type):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', port))
        s.listen(5)
        print(f"Server listening for {camera_type} camera on 0.0.0.0:{port}")
        while True:
            conn, addr = s.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr, camera_type))
            client_thread.start()

if __name__ == "__main__":
    threading.Thread(target=start_camera_server, args=(BACK_CAMERA_PORT, "Back")).start()
    threading.Thread(target=start_camera_server, args=(FRONT_CAMERA_PORT, "Front")).start()