import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}



def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

def save_keypoints_ankles(frame, keypoints_with_scores, confidence_threshold, df_ankles,frame_nb):
    df_ankles.loc[frame_nb,'Frame'] = frame_nb+1
    for person, keypoints in enumerate(keypoints_with_scores):
    
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

        left_ankle = shaped[15]
        right_ankle = shaped[16]

        left_ankle_y, left_ankle_x, left_ankle_conf = left_ankle
        right_ankle_y, right_ankle_x, right_ankle_conf = right_ankle
        if left_ankle_conf > confidence_threshold:
                df_ankles.loc[frame_nb,'left_ankle_'+str(person+1)] = [left_ankle_x, left_ankle_y]
        if right_ankle_conf > confidence_threshold:
                df_ankles.loc[frame_nb,'right_ankle_'+str(person+1)] = [right_ankle_x, right_ankle_y]

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

def draw_boxes(frame, boxes, confidence_threshold):
    ymin, xmin, ymax, xmax, score = boxes[0]
    # Convert the coordinates from relative to absolute values
    height, width, _ = frame.shape
    ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)
    
    if score > confidence_threshold:
        # Draw the bounding box around the person
        cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), (0, 255, 0), 2)

# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, boxes_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)
    for person in boxes_with_scores:
        draw_boxes(frame, person, confidence_threshold)


        

