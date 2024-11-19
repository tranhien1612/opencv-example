import cv2, time, math, queue, threading
import numpy as np
from djitellopy import Tello

def key_control(drone, key):
    if key == ord('w'):
        drone.move_forward(30)
    elif key == ord('s'):
        drone.move_back(30)
    elif key == ord('a'):
        drone.move_left(30)
    elif key == ord('d'):
        drone.move_right(30)

    elif key == ord('i'):
        drone.move_up(30)
    elif key == ord('k'):
        drone.move_down(30)
    elif key == ord('j'):
        drone.rotate_counter_clockwise(30)
    elif key == ord('l'):
        drone.rotate_clockwise(30)

    elif key == ord('n'):
        drone.land()
    elif key == ord('m'):
        drone.takeoff()

''' ----------------------------- Custom PID ----------------------------- '''
xPID, yPID, zPID = [0.21, 0, 0.1], [0.27, 0, 0.1], [0.0021, 0, 0.1]
pError, pTime, I = 0, time.time(), 0

def PIDController(PID, target, cVal, limit=[-100, 100], pTime=0, pError=0, I=0):
    """
    PIDController calculates the control value based on the PID algorithm.

    Args:
        PID (list): List of PID coefficients [P, I, D].
        img (numpy.ndarray): Input image.
        target (float): Target value.
        cVal (float): Current value.
        limit (list, optional): Control value limits. Defaults to [-100, 100].
        pTime (float, optional): Previous time. Defaults to 0.
        pError (float, optional): Previous error. Defaults to 0.
        I (float, optional): Integral term. Defaults to 0.
        draw (bool, optional): Flag to draw control value on image. Defaults to False.

    Returns:
        int: Control value.
    """
    t = time.time() - pTime
    error = target - cVal
    P = PID[0] * error
    I = I + (PID[1] * error * t)
    D = PID[2] * (error - pError) / t

    val = P + I + D
    val = float(np.clip(val, limit[0], limit[1]))

    pError = error
    pTime = time.time()

    return int(val)

''' ----------------------------- Custom PID ----------------------------- '''

''' -------------------------------- ROI  -------------------------------- '''
roi_box = ()
roiFlag = False
tracking = False

def mouse_handle(event, x, y, flags, param):
    global roi_box, roiFlag, tracking
    if event == cv2.EVENT_LBUTTONDOWN and roiFlag == False:
        roi_box = roi_box + (x, y,)
    elif event == cv2.EVENT_LBUTTONUP and roiFlag == False:
        # w = x - roi_box[0]
        # h = y - roi_box[1]
        roi_box = roi_box + (x - roi_box[0], y - roi_box[1], )
        roiFlag = True

''' -------------------------------- ROI  -------------------------------- '''
def process_obj(drone, frame, obj):
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    (x, y, w, h) = [int(v) for v in obj]
    obj_x, obj_y = (w // 2 + x), (h // 2 + y)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, (obj_x, obj_y), 5, (255, 255, 0), -1)
    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1) 
    cv2.line(frame, (obj_x, obj_y), (center_x, center_y), (0, 0, 255), 2)

    xVal = PIDController(xPID, center_x, obj_x)
    yVal = PIDController(yPID, center_y, obj_y)
    zVal = PIDController(zPID, (width*height)//2, w*h, limit=[-10, 20])
    print(f"x: {xVal}, y: {yVal}, z: {zVal}")
    drone.send_rc_control(0, zVal, yVal, -xVal)

def process_video(drone, fixROI=False):
    global roi_box, roiFlag, tracking
    frame_read = drone.get_frame_read()
    print(f"Image Size: {frame_read.frame.shape[0]}, {frame_read.frame.shape[1]}")

    if fixROI:
        roi_box = cv2.selectROI(frame_read.frame)
        # tracker = cv2.legacy.TrackerKCF_create()
        tracker = cv2.legacy.TrackerCSRT_create()
        ret = tracker.init(frame_read.frame, roi_box)
    else:
        ''' Select box while video is running '''
        cv2.imshow("Image", frame_read.frame)
        cv2.setMouseCallback("Image", mouse_handle)

    while True:
        frame = frame_read.frame

        if fixROI:
            ret, obj = tracker.update(frame)
            if ret == True:
                process_obj(drone, frame, obj)
            else:
                cv2.destroyAllWindows()
                drone.end()
        else:
            if roiFlag == True and tracking == False:
                tracker = cv2.legacy.TrackerCSRT_create()
                ret = tracker.init(frame, roi_box)
                tracking = True
        
            if tracking == True:
                ret, obj = tracker.update(frame)
                if ret == True:
                    process_obj(drone, frame, obj)
                else:
                    cv2.destroyAllWindows()
                    drone.end()

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xff
        key_control(drone, key)
        if key == 27:
            cv2.destroyAllWindows()
            drone.end()
            break

def process_video_fixROI(drone):
    frame_read = drone.get_frame_read()
    height, width, _ = frame_read.frame.shape
    center_x, center_y = width // 2, height // 2
    print(f"Image Size: {height}, {width}")

    bbox = cv2.selectROI(frame_read.frame)
    # tracker = cv2.legacy.TrackerKCF_create()
    tracker = cv2.legacy.TrackerCSRT_create()
    ret = tracker.init(frame_read.frame, bbox)

    while True:
        frame = frame_read.frame
        ret, bbox = tracker.update(frame)

        if ret == True:
            (x, y, w, h) = [int(v) for v in bbox]
            obj_x, obj_y = (w // 2 + x), (h // 2 + y)

            xVal = PIDController(xPID, center_x, obj_x)
            yVal = PIDController(yPID, center_y, obj_y)
            zVal = PIDController(zPID, (width*height)//2, w*h, limit=[-10, 20])
            drone.send_rc_control(0, zVal, yVal, -xVal)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (obj_x, obj_y), 5, (255, 255, 0), -1)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1) 
            cv2.line(frame, (obj_x, obj_y), (center_x, center_y), (0, 0, 255), 2)
        else:
            cv2.destroyAllWindows()
            drone.end()

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xff
        key_control(drone, key)
        if key == 27:
            cv2.destroyAllWindows()
            drone.end()
            break

def main():
    drone = Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}")
    drone.streamoff()
    drone.streamon()

    drone.takeoff()
    drone.move_up(100)
    
    recorder = threading.Thread(target=process_video, args=(drone, False,))
    recorder.start()

if __name__ == '__main__':
    main()


'''
def process_video(drone):
    global roi_box, roiFlag, tracking
    frame_read = drone.get_frame_read()
    height, width, _ = frame_read.frame.shape
    center_x, center_y = width // 2, height // 2
    print(f"Image Size: {height}, {width}")

 
    # bbox = cv2.selectROI(frame_read.frame)
    # # tracker = cv2.legacy.TrackerKCF_create()
    # tracker = cv2.legacy.TrackerCSRT_create()
    # ret = tracker.init(frame_read.frame, bbox)
    
    cv2.imshow("Image", frame_read.frame)
    cv2.setMouseCallback("Image", mouse_handle)

    while True:
        frame = frame_read.frame

        if roiFlag == True and tracking == False:
            tracker = cv2.legacy.TrackerCSRT_create()
            ret = tracker.init(frame, roi_box)
            tracking = True

        if tracking == True:
            ret, bbox = tracker.update(frame)
            if ret == True:
                (x, y, w, h) = [int(v) for v in bbox]
                obj_x, obj_y = (w // 2 + x), (h // 2 + y)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (obj_x, obj_y), 5, (255, 255, 0), -1)
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1) 
                cv2.line(frame, (obj_x, obj_y), (center_x, center_y), (0, 0, 255), 2)

                xVal = PIDController(xPID, center_x, obj_x)
                yVal = PIDController(yPID, center_y, obj_y)
                zVal = PIDController(zPID, (width*height)//2, w*h, limit=[-10, 20])
                drone.send_rc_control(0, zVal, yVal, -xVal)

            else:
                cv2.destroyAllWindows()
                drone.end()

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xff
        key_control(drone, key)
        if key == 27:
            cv2.destroyAllWindows()
            drone.end()
            break

'''