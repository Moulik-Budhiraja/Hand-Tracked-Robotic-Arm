import cv2
import mediapipe as mp
from dataclasses import dataclass
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import matplotlib.pyplot as plt
import numpy as np;
import math;

@dataclass
class FakeLandmark:
  x: float
  y: float
  z: float

def weightedSum(data, depth=10):
  weighted_sum = 0;
  denom = 2 ** depth;

  counter = 1;

  for i in data[-depth:]:
    weighted_sum += (2 ** counter) * i / denom;
    counter += 1;

  return weighted_sum;

def getAreaFromLandmarks(landmarks):

  min_x, max_x = landmarks[0].x, landmarks[0].x;
  min_y, max_y = landmarks[0].y, landmarks[0].y;

  marks = [0, 5, 9, 13, 17]

  for landmark in [landmarks[j] for j in marks]:
    
    if (landmark.x > max_x):
      max_x = landmark.x;
    if (landmark.x < min_x):
      min_x = landmark.x;
    if (landmark.y > max_y):
      max_y = landmark.y;
    if (landmark.y < min_y):
      min_y = landmark.y;

  return (min_x, min_y), (max_x, max_y), (max_x - min_x) * (max_y - min_y);
 
def transformScreenLandmarks(landmarks, image):
  return [FakeLandmark(landmark.x * image.shape[1], landmark.y * image.shape[0], 0) for landmark in landmarks];

def getAmountHandClosed(landmarks):
  vec1 = (landmarks[5].x - landmarks[0].x, landmarks[5].y - landmarks[0].y, landmarks[5].z - landmarks[0].z);
  vec2 = (landmarks[6].x - landmarks[5].x, landmarks[6].y - landmarks[5].y, landmarks[6].z - landmarks[5].z);

  return abs((vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]) / (math.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2) * math.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2)));

def getAmountHandTilted(landmarks):
  up_vec = (0, -1, 0);
  hand_vec = (landmarks[9].x - landmarks[0].x, landmarks[9].y - landmarks[0].y, landmarks[9].z - landmarks[0].z);
  
  return (up_vec[0] * hand_vec[0] + up_vec[1] * hand_vec[1] + up_vec[2] * hand_vec[2]) / (math.sqrt(up_vec[0] ** 2 + up_vec[1] ** 2 + up_vec[2] ** 2) * math.sqrt(hand_vec[0] ** 2 + hand_vec[1] ** 2 + hand_vec[2] ** 2))

def getX(landmark):
  return -1 * landmark.x;

def getY(landmark):
  return -landmark.y;

def getZ(landmark, offset):
  return landmark.z + offset;

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    max_num_hands=1,
    min_tracking_confidence=0.5) as hands:
  
  plt.ion();

  # Figure for distance graphs
  fig = plt.figure();
  ax = fig.add_subplot()

  line_screen, = ax.plot([], [], label="screen");
  line_world, = ax.plot([], [], label="world");
  line_div, = ax.plot([], [], label="combined");
  # line_smoothed, = ax.plot([], [], label="combined_smoothed");

  fig.legend(loc="upper left")

  # Figure for hand coordinate scatterplot
  fig_scatter = plt.figure()
  ax_scatter = fig_scatter.add_subplot(projection="3d")
  ax_scatter.view_init(elev=0, azim=0, roll=0)
  scatterplot = ax_scatter.scatter([], [], [], s=50)
  scatterline = ax_scatter.plot([], [], [])
  palmline = ax_scatter.plot([], [], [], color="red")
  thumbline = ax_scatter.plot([], [], [], color="black")
  indexline = ax_scatter.plot([], [], [], color="black")
  middleline = ax_scatter.plot([], [], [], color="black")
  ringline = ax_scatter.plot([], [], [], color="black")
  pinkyline = ax_scatter.plot([], [], [], color="black")

  ax_scatter.axes.set_xlim(-0.05, 0.15);
  ax_scatter.axes.set_ylim(-0.05, 0.15);
  ax_scatter.axes.set_zlim(-0.05, 0.15);

  # Initial data
  data_screen = [];
  data_world = [];
  data_div = [];
  data_smoothed = [];
  x_increment = [];

  # area_open = 9
  # area_closed = 10

  while cap.isOpened():

    success, image = cap.read()

    if not success:
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    min_p = (0, 0)
    max_p = (0, 0)
    smin_p = (0, 0)
    smax_p = (0, 0)

    if results.multi_hand_landmarks:

      min_p, max_p, world_area = getAreaFromLandmarks(results.multi_hand_world_landmarks[0].landmark);
      smin_p, smax_p, screen_area = getAreaFromLandmarks(transformScreenLandmarks(results.multi_hand_landmarks[0].landmark, image));

      screen_area /= 100000
      world_area *= 100

      # key = cv2.waitKey(5)
      # if key == ord("o"):
      #   print("Capture open hand")
      #   area_open = screen_area
      # elif key == ord("p"):
      #   print("Capture Closed Hand")
      #   area_closed = screen_area

      data_screen.append(screen_area);
      data_world.append(world_area);

      # World area / screen area ^ 3/4
      data_div.append(world_area / (screen_area ** 0.75));
      data_div[-1] -= data_div[-1] * 0.3 * ((1 - getAmountHandClosed(results.multi_hand_world_landmarks[0].landmark)) ** 2)# - 5 ** (1 - getAmountHandTilted(results.multi_hand_world_landmarks[0].landmark)));
      # data_div[-1] += data_div[-1] * 0.5 * (1 - getAmountHandTilted(results.multi_hand_world_landmarks[0].landmark) ** 2)

      print(1 - getAmountHandTilted(results.multi_hand_world_landmarks[0].landmark))
      # Weighted sum of last 20 datapoints in attempt to smooth out data
      data_smoothed.append(weightedSum(data_div, 20));
      
      x_increment.append(x_increment[-1] + 1 if len(x_increment) > 0 else 0);

      line_screen.set_xdata(x_increment);
      line_world.set_xdata(x_increment);
      line_div.set_xdata(x_increment);
      # line_smoothed.set_xdata(x_increment);
      
      line_screen.set_ydata(data_screen);
      line_world.set_ydata(data_world);
      line_div.set_ydata(data_div);
      # line_smoothed.set_ydata(data_smoothed);

      palm_indexes = []

      # x_scatter_data = list(map(getZ, results.multi_hand_world_landmarks[0].landmark))
      # y_scatter_data = list(map(getX, results.multi_hand_world_landmarks[0].landmark))
      # z_scatter_data = list(map(getY, results.multi_hand_world_landmarks[0].landmark))

      # # Plotting hand coordinates
      # scatterplot = (x_scatter_data, y_scatter_data, z_scatter_data)

      ax_scatter.axes.set_xlim(-0.05, 0.3);
      ax_scatter.axes.set_ylim(-0.05, 0.3);
      ax_scatter.axes.set_zlim(-0.05, 0.3);


      lines = [[(0, 5), (5, 9), (9, 13), (13, 17), (17, 0), (0, 5)],
               [(0, 1), (1, 2), (2, 3), (3, 4)],
               [(5, 6), (6, 7), (7, 8)],
               [(9, 10), (10, 11), (11, 12)],
               [(13, 14), (14, 15), (15, 16)],
               [(17, 18), (18, 19), (19, 20)]]
      
      x_palm_data = list(map(lambda landmark : getZ(landmark, data_div[-1] / 5), [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[0]]))
      y_palm_data = list(map(getX, [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[0]]))
      z_palm_data = list(map(getY, [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[0]]))
      x_thumb_data = list(map(lambda landmark : getZ(landmark, data_div[-1] / 5), [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[1]]))
      y_thumb_data = list(map(getX, [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[1]]))
      z_thumb_data = list(map(getY, [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[1]]))
      x_index_data = list(map(lambda landmark : getZ(landmark, data_div[-1] / 5), [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[2]]))
      y_index_data = list(map(getX, [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[2]]))
      z_index_data = list(map(getY, [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[2]]))
      x_middle_data = list(map(lambda landmark : getZ(landmark, data_div[-1] / 5), [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[3]]))
      y_middle_data = list(map(getX, [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[3]]))
      z_middle_data = list(map(getY, [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[3]]))
      x_ring_data = list(map(lambda landmark : getZ(landmark, data_div[-1] / 5), [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[4]]))
      y_ring_data = list(map(getX, [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[4]]))
      z_ring_data = list(map(getY, [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[4]]))
      x_pinky_data = list(map(lambda landmark : getZ(landmark, data_div[-1] / 5), [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[5]]))
      y_pinky_data = list(map(getX, [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[5]]))
      z_pinky_data = list(map(getY, [results.multi_hand_world_landmarks[0].landmark[start] for start, end in lines[5]]))

      palmline[0].set_data_3d(x_palm_data, y_palm_data, z_palm_data)
      thumbline[0].set_data_3d(x_thumb_data, y_thumb_data, z_thumb_data)
      indexline[0].set_data_3d(x_index_data, y_index_data, z_index_data)
      middleline[0].set_data_3d(x_middle_data, y_middle_data, z_middle_data)
      ringline[0].set_data_3d(x_ring_data, y_ring_data, z_ring_data)
      pinkyline[0].set_data_3d(x_pinky_data, y_pinky_data, z_pinky_data)

      ax.set_xbound(min(x_increment[-100:]), max(x_increment[-100:]))
      ax.set_ybound(0, max(map(max, data_screen[-100:], data_world[-100:], data_div[-100:])))#, data_screen[-100:], data_world[-100:]))]) #max(max(data_div[-100:]), max(data_screen[-100:]), max(data_world[-100:]))])
      plt.draw()
      plt.pause(0.001)

      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    
        image = cv2.rectangle(image, (int(smin_p[0]), int(smin_p[1])), (int(smax_p[0]), int(smax_p[1])), (0, 255, 0), 3)
        image = cv2.rectangle(image, (int(smin_p[0]), int(smin_p[1])), 
                              (int(smax_p[0]),  
                              int(smin_p[1] + (max_p[1] * image.shape[0] - min_p[1] * image.shape[0]) * (smax_p[0] - smin_p[0]) / (max_p[0] * image.shape[1] - min_p[0] * image.shape[1])))
                              , (255, 0, 0), 3)
        image = cv2.rectangle(image, (int(smin_p[0]), int(smin_p[1])), 
                              (int(smin_p[0] + (max_p[0] * image.shape[1] - min_p[0] * image.shape[1])), 
                              int(smin_p[1] + (max_p[1] * image.shape[0] - min_p[1] * image.shape[0])))
                              , (255, 0, 0), 3)
    image = cv2.flip(image, 1)

    # image = cv2.putText(image, f"Open Hand: {area_open}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,  0, 0), 2, cv2.LINE_AA)
    # image = cv2.putText(image, f"Closed Hand: {area_closed}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,  0, 0), 2, cv2.LINE_AA)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()