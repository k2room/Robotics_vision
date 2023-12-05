import cv2                                # state of the art computer vision algorithms library
import math
import numpy as np                        # fundamental package for scientific computing
from IPython.display import clear_output  # Clear the screen
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import serial
print("Environment Ready")

def serial_connect():
    # ser_name = '/dev/ttyACM0'
    ser_name = '/dev/ttyACM0'
    try:
        try:
            ser = serial.Serial(ser_name, 57600)
        except:
            serial.Serial(ser_name, 57600).close()
            ser = serial.Serial(ser_name,57600,timeout=0.1) 
        finally:
            print('ser:', ser)
            return ser
    except:
        print('Serial error')
        return "no serial"
    

def relative_distance(dist, pos, FOV, stream_size):
    fov_degrees = FOV[0]
    resolution_width = stream_size[0]
    angle_per_pixel = fov_degrees / resolution_width
    dif = pos - resolution_width/2 # 화면상에서 목적지 위치와 로봇 시점 가운데의 차이
    dif = dif*angle_per_pixel
    real_pos = dist*math.tan(math.radians(dif))

    return [dist, real_pos]

def send_msg(dist, pos, start, last_turn):
    threshold = 60
    # turn left until detect object
    if start and dist == -777 and pos == -777:
        state = 'k'
    # turn left untsil detect object
    elif dist == -777 and pos == -777 and (last_turn=='k' or last_turn=='l'):
        state = 'e'
    elif dist == -777 and pos == -777 and (last_turn=='e' or last_turn=='r'):
        state = 'k'
    # straight
    elif dist >= 150 and abs(pos) <= 120:
        state = 'g'
    # straight
    elif dist < 150 and dist >= 20 and abs(pos) <= 60:
        state = 'g'

    # turn left
    elif dist >= 150 and pos < -120:
        state = 'l'
    # turn right
    elif dist >= 150 and pos > 120:
        state = 'r'
    # turn left
    elif dist < 150 and pos < -60:
        state = 'l'
    # turn right
    elif dist < 150 and pos > 60:
        state = 'r'
    # stop
    elif dist < 20:
        state = 'x'
    
    return state

def stream(ser, use_window):

    FPS = 30
    stream_size = [640, 480]
    FOV = [87, 58]
    msg_period = 5

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    print("device_product_line: ", device_product_line)

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # Steam type, size, format
    """
        가능한 사이즈 및 fps 

    """
    config.enable_stream(rs.stream.depth, stream_size[0], stream_size[1], rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, stream_size[0], stream_size[1], rs.format.bgr8, FPS)

    # Start streaming
    pipeline.start(config)

    try:
        start = True
        msg_seq = []
        last_turn = ''
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            
            """
                RGB & Depth stream code + Red mask
            """
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Red color mask
            lower_red = np.array([45, 20, 105]) # 빨간색 범위의 하한
            upper_red = np.array([70, 35, 255]) # 빨간색 범위의 상한
            red_mask = cv2.inRange(color_image, lower_red, upper_red)

            # 모폴로지 연산으로 노이즈 제거
            kernel = np.ones((3,3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

            # Visualize depth of red area
            red_image = cv2.bitwise_and(color_image, color_image, mask=red_mask)
            red_depth = cv2.bitwise_and(depth_colormap, depth_colormap, mask=red_mask)

            # 깊이 이미지에서 빨간색 영역에 해당하는 깊이 데이터 추출
            red_depth_values = depth_image[red_mask != 0]

            # 빨간색 영역의 중심 계산
            red_y, red_x = np.where(red_mask != 0)
            if red_x.size > 0 and red_y.size > 0:
                cX = int(np.mean(red_x))
                cY = int(np.mean(red_y))
                cv2.circle(color_image, (cX, cY), 3, (255, 255, 255), -1)
                cv2.circle(red_image, (cX, cY), 3, (255, 255, 255), -1)

            # 깊이 데이터의 평균 계산
            if red_depth_values.size > 0:
                average_depth = np.mean(red_depth_values)
                coord = relative_distance(average_depth, cX, FOV, stream_size)

                # print("Average Depth of Red Area: {:.2f}mm".format(average_depth), end='\r')
                # print("Relative Coordinates(cm): {:.2f} (distance), {:.2f} (position)".format(coord[0]/10, coord[1]/10), end='\r')
                # print("Relative Coordinates(cm): {:.2f} (distance), {:.2f} (position)".format(coord[0]/10, coord[1]/10))
                message = send_msg(coord[0]/10, coord[1]/10, start, last_turn)
                # if (message == 'k') or (message == 'e') or (message == 'l') or (message == 'r'):
                if coord[1]/10 > 0:
                    last_turn = 'k'
                else:
                    last_turn = 'e'
                start = False
            else:
                # print("Average Depth of Red Area: No red area detected.", end='\r')
                print("Average Depth of Red Area: No red area detected. last_turn:", last_turn)
                message = send_msg(-777, -777, start, last_turn)
            msg_seq.append(message)

            if len(msg_seq) == msg_period:
                print("SEND MSG:", max(msg_seq).encode('utf-8'))

                if ser !='':
                    ser.write(max(msg_seq).encode('utf-8'))
                msg_seq = []

            # 이미지 크기 조정 및 결합
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                resized_red_image = cv2.resize(red_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap, resized_red_image, red_depth))
            else:
                images = np.hstack((color_image, depth_colormap, red_image, red_depth))
                # images = np.vstack((np.hstack((color_image, depth_colormap)), np.hstack((red_image, red_depth))))

            # Show images
            if use_window:
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)

            if cv2.waitKey(1) & 0xFF == 27: # ESC 키
                break

    finally:
        # Stop streaming
        pipeline.stop()
        if use_window:
            cv2.destroyWindow('RealSense')
    

if __name__=="__main__":
    use_serial = True
    use_window = True
    if use_serial:
        ser = serial_connect()
        while ser == "no serial":
            ser = serial_connect()
            print("serial:", ser)
        stream(ser, use_window)
    else:
        ser = ''
        stream(ser, use_window)