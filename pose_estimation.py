import cv2
import math
import numpy as np
import sys
import os
import dlib
import glob
import face_utils

predictor_path = "shape_predictor_68_face_landmarks.dat"
cap = cv2.VideoCapture("bilateralLoss.MOV")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
findex = -1
nose_difference = []


  

while (True):

    
    

    _,frame = cap.read()
    findex += 1

    #dets = detector(frame, 1)
    #for k, d in enumerate(dets):
        #shape = predictor(frame, d) 

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray)
    for det in dets:
       shape = predictor(gray, det)
       #shape1 = face_utils.shape_to_np(shape)
            
    r = 0
    g = 100
    b = 200
    index = 0
    landmark_list = {}
    
    
    for i in shape.parts():
        cords = (int(str(i).split(",")[0][1:]),int(str(i).split(",")[1][:-1]))
        image = cv2.circle(frame, cords, 2, (b,g,r), -1)
        #cv2.putText(image, str(index),cords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
        #print(f"Index {index} :  {cords}")
        landmark_list[str(index)] = cords
        index +=1
        r +=5
        g +=5
        b +=5
    
    size = frame.shape #(height, width, color_channel)

    """
    landmark_Cord = [] 
    #landmark_Cord.append(findex)
    landmark_Cord.append(shape.part(48).x)
    landmark_Cord.append(shape.part(48).y)
    print (landmark_Cord)
    """
       


    image_points = np.array([
                            landmark_list["30"], # Nose
                            landmark_list["8"],  # Chin
                            landmark_list["45"], # Left eye left corner
                            landmark_list["36"], # Right eye right cornee
                            landmark_list["54"], # Left Mouth corner
                            landmark_list["48"]  # Right mouth corner
                        ], dtype="double")

    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])
 
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    # (x,y) of p by projecting P

    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    """
    projectpoint_cord = []
    #projectpoint_cord.append(findex)
    projectpoint_cord.append(int (modelpts[5][0][0]))
    projectpoint_cord.append(int (modelpts[5][0][1]))
    print (projectpoint_cord)
    """

    """
    difference = [] 
    difference.append(findex)
    difference.append(abs ( (shape.part(30).x) - (int (modelpts[0][0][0]))))
    difference.append(abs ( (shape.part(30).y) - (int (modelpts[0][0][1]))))
    print (difference)
    """
    #print (abs ( (shape.part(48).y) - (int (modelpts[5][0][1]))))
    
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    # concatenate 
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] # ie pitch yaw roll angles

    #convert from degree value into radians
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    # convert angle radian value to degree
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    rotate_degree = ((int(roll)), (int(pitch)), (int(yaw)))
    nose = (landmark_list["4"], landmark_list["5"])
        

    #imgpts,  modelpts, rotate_degree, nose = face_orientation(frame, landmark_list)

    #end_point: It is the ending coordinates of line. The coordinates are represented as tuples of two values 
    # i.e. (X coordinate value, Y coordinate value).
    cv2.line(frame, landmark_list["30"], tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
    cv2.line(frame, landmark_list["30"], tuple(imgpts[0].ravel()), (255,0,), 3) #BLUE
    cv2.line(frame, landmark_list["30"], tuple(imgpts[2].ravel()), (0,0,255), 3) #RED

    cv2.putText(frame, "FIndex:  " + str(findex), (20, 190), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,0), 1)
    #remapping = [2,3,0,4,5,1]
    for j in range(len(rotate_degree)):
        cv2.putText(frame, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
    h,w,_ = frame.shape
    

    cv2.putText(frame, "FIndex:  " + str(findex), (20, 190), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,0), 1)
    print (str(findex) + " " + str(rotate_degree[1]) + " " + str(rotate_degree[0])+ " " + str(rotate_degree[2]) )

    #print (str(findex) + " " + str(cord) + " " + str(rotate_degree[0])+ " " + str(rotate_degree[2]) )

    
    
    
    cv2.imshow("frame", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        cap.release()
        break
        
        
cv2.destroyAllWindows()
    





