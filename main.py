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

  for landmark in landmarks[1:]:
    
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

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    max_num_hands=1,
    min_tracking_confidence=0.5) as hands:
  
  plt.ion();
  
  line_screen, = plt.plot([], [], label="screen");
  line_world, = plt.plot([], [], label="world");
  line_div, = plt.plot([], [], label="combined");
  line_smoothed, = plt.plot([], [], label="combined_smoothed");

  plt.legend(loc="upper left")
  
  data_screen = [0];
  data_world = [0];
  data_div = [0];
  data_smoothed = [0];
  x_increment = [0];

  while cap.isOpened():

    success, image = cap.read()

    if not success:
      continue

    # To improve performanceb, optionally mark the image as not writeable to
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

      data_screen.append(screen_area / 100000);
      data_world.append(world_area * 100);

      # World area / screen area ^ 3/4
      data_div.append(world_area / (screen_area ** 0.75));

      # Weighted sum of last 20 datapoints in attempt to smooth out data
      data_smoothed.append(weightedSum(data_div, 20));
      
      x_increment.append(x_increment[-1] + 1);

      # line_screen.set_xdata(x_increment);
      # line_world.set_xdata(x_increment);
      line_div.set_xdata(x_increment);
      line_smoothed.set_xdata(x_increment);
      
      # line_screen.set_ydata(data_screen);
      # line_world.set_ydata(data_world);
      line_div.set_ydata(data_div);
      line_smoothed.set_ydata(data_smoothed);

      plt.axis([max(0, x_increment[-1] - 100), x_increment[-1], 0, max(map(max, data_smoothed[-100:], data_div[-100:]))])#, data_screen[-100:], data_world[-100:]))]) #max(max(data_div[-100:]), max(data_screen[-100:]), max(data_world[-100:]))])
      plt.draw()

      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    
    image = cv2.rectangle(image, (int(smax_p[0] * image.shape[1]), int(smax_p[1] * image.shape[0])), (int(smin_p[0] * image.shape[1]), int(smin_p[1] * image.shape[0])), (0, 255, 0), 3)
    image = cv2.rectangle(image, (int(max_p[0] * image.shape[1]), int(max_p[1] * image.shape[0])), (int(min_p[0] * image.shape[1]), int(min_p[1] * image.shape[0])), (255, 0, 0), 3)
    image = cv2.flip(image, 1)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()