import cv2                                # state of the art computer vision algorithms library
import math
import numpy as np                        # fundamental package for scientific computing
from IPython.display import clear_output  # Clear the screen
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import serial
print("Environment Ready")


# Aux.
deg2rad = np.pi/180.0
G_mem = 0
dst_p_mem = 1.0
R_mem = 0   # has to be zero if the red marker is not initially given
LMT_ENB = 0



def DRCTR(off_g, dst_g, flg_g, off_p, dst_p, flg_p):
    global deg2rad, G_mem, dst_p_mem, R_mem, LMT_ENB
    # ||| ---------- director settings ---------- |||
    # NAV. tune
    ini_rot = 'L'   # initial rotation direction to find red marker
    #ini_rot = 'R'

    # param. when there is no goal
    tht_g_CRT1 = deg2rad*45.0
    tht_g_CRT2 = deg2rad*15.0
    mod_tht_g_CRT = 0.4     # metre

    # general avoidance settings
    r_DIR_pass = 1.2
    mod_r_DIR1 = deg2rad*90
    mod_r_DIR2 = 0.0001
    tht_DIR_pass = deg2rad*85.0

    # limiter settings
    dst_diff_CRIT = 0.3   # metre
    LMT_stp = 3
    LMT_r = 0.20    # metre
    dst_EMR = 0.35   # metre

    # discrete radius decider settings
    global r_CRT
    r_CRT = np.array([2.0, 1.5, 0.7, 0.5, 0.35, 0.3, 0.25])  # metre
    gamma_std = r_CRT[0]     # metre
    gamma = 0.7     # < 1.0 for safety



    # VC_set index
    global idx_st, \
    \
        idx_prl_L15, idx_prl_R15, \
        idx_prl_L30, idx_prl_R30, \
        idx_prl_L45, idx_prl_R45, \
        idx_prl_L60, idx_prl_R60, \
    \
        idx_arc_L250, idx_arc_R250, \
        idx_arc_L300, idx_arc_R300, \
        idx_arc_L350, idx_arc_R350, \
        idx_arc_L500, idx_arc_R500, \
        idx_arc_L700, idx_arc_R700, \
        idx_arc_L1500, idx_arc_R1500, \
    \
        idx_arc_L_LMT, idx_arc_R_LMT, \
    \
        idx_rot_L15, idx_rot_R15

    idx_st = 'g'

    idx_prl_L30 = 'g'
    idx_prl_R30 = 'g'

    idx_arc_L250 = 'w'
    idx_arc_R250 = 'q'
    idx_arc_L300 = 'j'
    idx_arc_R300 = 'd'
    idx_arc_L350 = 'y'
    idx_arc_R350 = 't'
    idx_arc_L500 = 's'
    idx_arc_R500 = 'h'
    idx_arc_L700 = 'k'
    idx_arc_R700 = 'f'
    idx_arc_L1500 = 'u'
    idx_arc_R1500 = 'v'

    idx_arc_L_LMT = idx_arc_L250
    idx_arc_R_LMT = idx_arc_R250

    idx_rot_L15 = 'n'
    idx_rot_R15 = 'm'


    # ||| ---------- end of director settings ---------- |||



    # ----- pre-processing -----
    r_CRT = gamma_std*np.power( (r_CRT / gamma_std), gamma )

    if ( flg_g == 1 ):
        tht_g = np.arctan2(off_g, dst_g)
        if ( tht_g > 0 ):
            R_mem = 1
        else:
            R_mem = -1

        tht_g_CRT1 = tht_g_CRT1*( 1.0 - np.exp( (-1.0/mod_tht_g_CRT)*dst_g ) )
        tht_g_CRT2 = tht_g_CRT2*( 1.0 - np.exp( (-1.0/mod_tht_g_CRT)*dst_g ) )

    if ( flg_p == 1 ) and ( R_mem != 0 ):
        tht_p = np.arctan2(off_p, dst_p)
        r_DIR_pass = r_DIR_pass*( 1.0 - np.exp( (abs(tht_p) - mod_r_DIR1) / mod_r_DIR2 ) )
        dst_pass = dst_p - r_DIR_pass*np.cos(tht_DIR_pass)

        dst_diff = dst_p - dst_p_mem
        if ( dst_diff > dst_diff_CRIT ) or ( G_mem == 0 ):
            LMT_ENB = LMT_stp
            if ( R_mem > 0 ):
                G_mem = 1
            else:
                G_mem = -1
        dst_p_mem = dst_p

        if ( G_mem > 0 ):
            off_pass = off_p + r_DIR_pass*np.sin(tht_DIR_pass)
            flg_pass = 1
        else:
            off_pass = off_p - r_DIR_pass*np.sin(tht_DIR_pass)
            flg_pass = -1

        tht_pass = np.arctan2(off_pass, dst_pass)
        flg_pass = flg_pass*tht_pass

        r_pass = ( off_pass**2 + dst_pass**2 ) / (2*off_pass)
        if ( LMT_ENB > 0 ) and ( dst_p > dst_EMR ):
            if ( abs(r_pass) < LMT_r ):
               r_pass = np.sign(r_pass)*LMT_r
            LMT_ENB -= 1
        print('')
        print('R_PASS:', r_pass)
        print('')

    


    # ----- decision making -----
    # goal location has not been identified
    if ( R_mem == 0 ):
        if ( ini_rot == 'L' ):
            return idx_rot_L15
        else:
            return idx_rot_R15

    # general obstacle avoidance
    elif ( flg_p == 1 ):
        return obs_AVD(r_pass, flg_pass)

    # towards goal without obstacle
    elif ( flg_g == 1 )and ( flg_p == 0 ) :

        if ( abs(tht_g) < tht_g_CRT2 ):
            return idx_st

        elif ( abs(tht_g) < tht_g_CRT1 ) or ( LMT_ENB > 0 ):
            if (LMT_ENB > 0):
                LMT_ENB -= 1

            if ( tht_g > 0 ):
                return idx_arc_R_LMT
            else:
                return idx_arc_L_LMT

        else:
            if ( tht_g > 0 ):
                return idx_rot_R15
            else:
                return idx_rot_L15

    else:
        if ( R_mem != 0 ):
            if (LMT_ENB > 0):
                LMT_ENB -= 1
                if ( R_mem > 0 ):
                    return idx_arc_R_LMT
                else:
                    return idx_arc_L_LMT
                
            else:
                if ( R_mem > 0 ):
                    return idx_rot_R15
                else:
                    return idx_rot_L15
        
        else:
            if ( ini_rot == 'L' ):
                return idx_rot_L15
            else:
                return idx_rot_R15
            
            

def obs_AVD(r_pass, flg_pass):
        # criterion for going straight
        if ( abs(r_pass) > r_CRT[0] ) or ( flg_pass < 0 ):
               return idx_st

        # criteria for arcs
        elif ( abs(r_pass) > r_CRT[1] ):
            if ( r_pass > 0 ):
                return idx_arc_R1500
            else:
                return idx_arc_L1500

        elif ( abs(r_pass) > r_CRT[2] ):
            if ( r_pass > 0 ):
                return idx_arc_R700
            else:
                return idx_arc_L700

        elif ( abs(r_pass) > r_CRT[3] ):
            if ( r_pass > 0 ):
                return idx_arc_R500
            else:
                return idx_arc_L500
            
        elif ( abs(r_pass) > r_CRT[4] ):
            if ( r_pass > 0 ):
                return idx_arc_R350
            else:
                return idx_arc_L350

        elif ( abs(r_pass) > r_CRT[5] ):
            if ( r_pass > 0 ):
                return idx_arc_R300
            else:
                return idx_arc_L300
            
        elif ( abs(r_pass) > r_CRT[6] ):
            if ( r_pass > 0 ):
                return idx_arc_R250
            else:
                return idx_arc_L250
            
        # rotation
        else:
            if r_pass > 0:
                return idx_rot_R15
            else:
                return idx_rot_L15
            


def DRCTR_1(off_g, dst_g, flg_g, off_p, dst_p, flg_p):
    # ||| ---------- director settings ---------- |||
    # NAV. tune
    tht_g_CRT1 = deg2rad*50.0
    tht_g_CRT2 = deg2rad*15.0
    mod_tht_g_CRT = 0.4 # metre

    r_DIR_pass = 0.85
    mod_r_DIR = 0.0000000000001 # metre
    tht_DIR_pass = deg2rad*90.0

    tht_p_DECI = deg2rad*0.0
    tht_g_DECI = deg2rad*0.0

    tht_CRT = np.array([12.0, 15.0, 45.0+0.0, 45.0+2.5, 60.0-1])
    gamma = 1.0

    # VC_set index
    # ! if new index is added, add it also in the idx_set
    idx_st = 'g'

    ''''''
    idx_prl_L15 = 'g'
    idx_prl_R15 = 'g'
    idx_prl_L30 = 'g'
    idx_prl_R30 = 'g'
    idx_prl_L45 = 'y'
    idx_prl_R45 = 't'
    idx_prl_L60 = 'i'
    idx_prl_R60 = 'u'
    idx_prl_L75 = 'p'
    idx_prl_R75 = 'o'

    idx_arc_L250 = 'h'
    idx_arc_R250 = 's'
    idx_arc_L300 = 'j'
    idx_arc_R300 = 'd'
    idx_arc_L1500 = 'k'
    idx_arc_R1500 = 'f'

    idx_rot_L15 = 'n'
    idx_rot_R15 = 'm'
    ''''''

    '''
    idx_prl_L15 = 'w'
    idx_prl_R15 = 'q'
    idx_prl_L30 = 'r'
    idx_prl_R30 = 'e'
    idx_prl_L45 = 'y'
    idx_prl_R45 = 't'
    idx_prl_L60 = 'i'
    idx_prl_R60 = 'u'
    idx_prl_L75 = 'p'
    idx_prl_R75 = 'o'

    idx_arc_L250 = 'h'
    idx_arc_R250 = 's'
    idx_arc_L300 = 'j'
    idx_arc_R300 = 'd'
    idx_arc_L1500 = 'k'
    idx_arc_R1500 = 'f'

    idx_rot_L15 = 'n'
    idx_rot_R15 = 'm'
    '''

    idx_set = [idx_st, idx_prl_L15, idx_prl_R15, idx_prl_L30, idx_prl_R30, idx_prl_L45, idx_prl_R45, idx_prl_L60, idx_prl_R60, idx_arc_L300, idx_arc_R300, idx_rot_L15, idx_rot_R15]
    # ||| ---------- end of director settings ---------- |||



    # ----- pre-processing -----
    tht_CRT_use = deg2rad*( 90.0*np.power( (tht_CRT/90.0), gamma ) )

    if ( flg_g == 1 ):
        tht_g = np.arctan2(off_g, dst_g)
        tht_g_CRT1 = tht_g_CRT1*( 1.0 - np.exp( (-1.0/mod_tht_g_CRT)*dst_g ) )
        tht_g_CRT2 = tht_g_CRT2*( 1.0 - np.exp( (-1.0/mod_tht_g_CRT)*dst_g ) )

    if ( flg_p == 1 ):
        tht_p = np.arctan2(off_p, dst_p)
        r_DIR_pass = r_DIR_pass*( 1.0 - np.exp( (-1.0/mod_r_DIR)*dst_p ) )
        dst_pass = dst_p - r_DIR_pass*np.cos(tht_DIR_pass)
        if ( abs(tht_p) < tht_p_DECI ): # if the obstacle is almost aligned with heading(not decisive)
            if ( flg_g == 1 ):
                if ( abs(tht_g) > tht_g_DECI ): # if the goal position is decisive enough,
                    if ( tht_g < 0 ):
                        off_pass = off_p - r_DIR_pass*np.sin(tht_DIR_pass)
                        flg_pass = -1
                    else:
                        off_pass = off_p + r_DIR_pass*np.sin(tht_DIR_pass)
                        flg_pass = 1
                else:
                    off_pass = off_p + r_DIR_pass*np.sin(tht_DIR_pass)
                    flg_pass = 1
            else:
                off_pass = off_p + r_DIR_pass*np.sin(tht_DIR_pass)
                flg_pass = 1

        else:
            if ( tht_p > 0 ):
                off_pass = off_p - r_DIR_pass*np.sin(tht_DIR_pass)
                flg_pass = -1
            else:
                off_pass = off_p + r_DIR_pass*np.sin(tht_DIR_pass)
                flg_pass = 1

        tht_pass = np.arctan2(off_pass, dst_pass)
        flg_pass = flg_pass*tht_pass



    # ----- decision making -----
    # goal and obstacle
    if ( flg_g and flg_p ):

        if ( abs(tht_g) >= tht_g_CRT1 ):
            if ( tht_g > 0) :
                return idx_rot_R15
            else:
                return idx_rot_L15

        else:
            return obs_AVD(tht_pass, tht_CRT_use, flg_pass, idx_set)


    # goal, but no obstacle
    elif ( (flg_g == 1) and (flg_p == 0) ):

        if ( abs(tht_g) < tht_g_CRT2 ):
            return idx_st

        elif ( abs(tht_g) < tht_g_CRT1 ):
            if tht_g > 0:
                return idx_arc_R300
            else:
                return idx_arc_L300

        else:
            if tht_g > 0:
                return idx_rot_R15
            else:
                return idx_rot_L15


    # no goal, but obstacle
    elif ( (flg_g == 0) and (flg_p == 1) ):
        return obs_AVD(tht_pass, tht_CRT_use, flg_pass, idx_set)


    # no goal, no obstacle
    else:
        return idx_arc_R300


def obs_AVD_1(tht_pass, tht_CRT_use, flg_pass, idx_set):
        if ( ( abs(tht_pass) < tht_CRT_use[0] ) or (flg_pass < 0) ):
               return idx_set[0]

        elif ( abs(tht_pass) < tht_CRT_use[1] ):
            if tht_pass > 0:
                return idx_set[2]
            else:
                return idx_set[1]

        elif ( abs(tht_pass) < tht_CRT_use[2] ):
                if tht_pass > 0:
                    return idx_set[4]
                else:
                    return idx_set[3]

        elif ( abs(tht_pass) < tht_CRT_use[3] ):
            if tht_pass > 0:
                return idx_set[6]
            else:
                return idx_set[5]

        elif ( abs(tht_pass) >= tht_CRT_use[4] ):
            if tht_pass > 0:
                return idx_set[8]
            else:
                return idx_set[7]

        else:
            if tht_pass > 0:
                return idx_set[12]
            else:
                return idx_set[11]

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

def send_msg(dist, pos, start, last_turn, obj_dist, obj_pos, ex_red, ex_green):
    off_g = pos
    dst_g = dist
    flg_g = ex_red
    off_p = obj_pos
    dst_p = obj_dist
    flg_p = ex_green
    print(off_g, dst_g, flg_g, off_p, dst_p, flg_p)
    state = DRCTR(off_g, dst_g, flg_g, off_p, dst_p, flg_p)

    return state

def send_msg_origin(dist, pos, start, last_turn):
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
        ex_green = False
        msg_seq = []
        last_turn = ''
        p = 30
        n_obj = 1
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
            lower_red = np.array([20, 0, 85]) # 빨간색 범위의 하한
            upper_red = np.array([90, 35, 255]) # 빨간색 범위의 상한
            red_mask = cv2.inRange(color_image, lower_red, upper_red)

            # Green color mask
            lower_green = np.array([50, 95, 0])
            upper_green = np.array([120, 200, 75]) 
            green_mask = cv2.inRange(color_image, lower_green, upper_green)

            # 모폴로지 연산으로 노이즈 제거
            kernel = np.ones((3,3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

            # Visualize depth of red area
            red_image = cv2.bitwise_and(color_image, color_image, mask=red_mask)
            # red_depth = cv2.bitwise_and(depth_colormap, depth_colormap, mask=red_mask)
            green_image = cv2.bitwise_and(color_image, color_image, mask=green_mask)
            # green_depth = cv2.bitwise_and(depth_colormap, depth_colormap, mask=green_mask)

            # 깊이 이미지에서 빨간색 영역에 해당하는 깊이 데이터 추출
            red_depth_values = depth_image[red_mask != 0]
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(green_mask, 8, cv2.CV_32S)


            object_areas = [(i, area) for i, (x, y, w, h, area) in enumerate(stats) if i != 0 and area > p]
            object_areas.sort(key=lambda x: x[1], reverse=True)  # Sort by area in descending order

            for i, area in object_areas[:n_obj]:
                x, y, w, h = stats[i][:4]
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(green_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                centroid = tuple(map(int, centroids[i]))
                cv2.circle(color_image, centroid, 3, (255, 255, 255), -1)
                cv2.circle(green_image, centroid, 3, (255, 255, 255), -1)
            
            if object_areas:
                # Get the largest green object's index
                largest_object_index = object_areas[0][0]

                # Create a mask for the largest green object
                largest_object_mask = (labels == largest_object_index)

                # Extract depth values for the largest green object
                largest_object_depth_values = depth_image[largest_object_mask]

                # Calculate the average depth
                if largest_object_depth_values.size > 0:  # Check if there are any depth values
                    obj_average_depth = np.mean(largest_object_depth_values)
                    obj_coord = relative_distance(obj_average_depth, centroid[0], FOV, stream_size)
                ex_green = True
            else:
                ex_green = False

            # 빨간색 영역의 중심 계산
            red_y, red_x = np.where(red_mask != 0)
            if red_x.size > 0 and red_y.size > 0:
                cX = int(np.mean(red_x))
                cY = int(np.mean(red_y))
                cv2.circle(color_image, (cX, cY), 3, (255, 255, 255), -1)
                cv2.circle(red_image, (cX, cY), 3, (255, 255, 255), -1)

            # 깊이 데이터의 평균 계산
            if red_depth_values.size > 0:
                ex_red = True
                average_depth = np.mean(red_depth_values)
                coord = relative_distance(average_depth, cX, FOV, stream_size)

                # print("Average Depth of Red Area: {:.2f}mm".format(average_depth), end='\r')
                # print("Relative Coordinates(cm): {:.2f} (distance), {:.2f} (position)".format(coord[0]/10, coord[1]/10), end='\r')
                # print("Relative Coordinates(cm): {:.2f} (distance), {:.2f} (position)".format(coord[0]/10, coord[1]/10))
                # message = send_msg(coord[0]/10, coord[1]/10, start, last_turn, obj_coord[0]/10, obj_coord[1]/10)
                message = send_msg(coord[0]/1000, coord[1]/1000, start, last_turn, obj_coord[0]/1000, obj_coord[1]/1000, ex_red, ex_green)
                # if (message == 'k') or (message == 'e') or (message == 'l') or (message == 'r'):
                if coord[1]/1000 > 0:
                    last_turn = 'k'
                else:
                    last_turn = 'e'
                start = False
            else:
                # print("Average Depth of Red Area: No red area detected.", end='\r')
                ex_red = False
                print("Average Depth of Red Area: No red area detected. last_turn:", last_turn)
                message = send_msg(-777, -777, start, last_turn, obj_coord[0]/1000, obj_coord[1]/1000, ex_red, ex_green)
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
                resized_green_image = cv2.resize(green_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                # resized_green_image = cv2.resize(isolated_green, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap, resized_green_image, resized_red_image))
            else:
                images = np.hstack((color_image, depth_colormap, green_image, red_image))
                # images = np.hstack((color_image, depth_colormap,isolated_green, red_image))
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
    #use_serial = False
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