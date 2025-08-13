import cv2
import time
import random
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(False, 1, 1, 0.3)

prev_frame_time = 0                   # time of the previous frame
curr_frame_time = 0                   # time of the current frame
tips_pts = np.array([[]], np.int32)   # numpy array of coordinates of the fingertips
Draw_pts = np.array([[]], np.int32)
colour = (255, 0, 0)
is_Draw_curr_Frame = False
is_Draw_prev_Frame = False

Color_Circle = {
  "Black": {
      "Center": (40, 40),
      "Radius": 20,
      "Color": (0, 0, 0),
      "is Active": False,
      "Drawing": [np.array([[]], np.int32)],
      "Distance": 300}
}


 #this function creates a bounding box around the hand
 #it takes as an argument the landmarks (0 ,4 , 8 , 12 ,16 , 20 represent the finger tips)
def Bounding_box_coords(lms):
  b_x1, b_y2, b_x2, b_y2 = (0, 0, 0, 0)

  b_y1 = min(lms[20].y, lms[16].y, lms[12].y, lms[8].y, lms[4].y, lms[0].y)
  b_y1 = int(b_y1 * h)

  b_y2 = max(lms[20].y, lms[16].y, lms[12].y, lms[8].y, lms[4].y, lms[0].y)
  b_y2 = int(b_y2 * h)

  b_x1 = min(lms[20].x, lms[16].x, lms[12].x, lms[8].x, lms[4].x, lms[0].x)
  b_x1 = int(b_x1 * w)

  b_x2 = max(lms[20].x, lms[16].x, lms[12].x, lms[8].x, lms[4].x, lms[0].x)
  b_x2 = int(b_x2 * w)
  # print(b_x1, b_x2)
  return (b_x1, b_y1), (b_x2, b_y2)

# this function calculates the distance between 2d-points
def distance(a, b):
  return (int(math.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))))

# this function determines whether the users hand is in draw position
# a hand is in  draw position when the tip of the index and the tip of the thumb are really close
# however that distance varies with the hand's closeness to the cam so we need to make it normalized with respect to a reference distance
# in our case the reference distance is between thumb tip and thumb dip
def Is_in_Draw_Position(handlms, w, h):
  thumb_tip_coords = (handlms[4].x * w, handlms[4].y * h)
  index_tip_coords = (handlms[8].x * w, handlms[8].y * h)
  thumb_dip_coords = (handlms[3].x * w, handlms[3].y * h)
  # index_dip_coords = (handlms[7].x * w, handlms[7].y * h)
  ref_d = distance(thumb_tip_coords, thumb_dip_coords)
  if (ref_d == 0):
      pass
  else:
      d = distance(thumb_tip_coords, index_tip_coords)
      final_d = int(d / ref_d)

  cv2.putText(img, str(final_d), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 3)
  if final_d < 1:
      return True
  return False


cap = cv2.VideoCapture(0)  # we set our pc webcam as our input
while cap.isOpened():  # while the webcam is opened
    ok, img = cap.read()  # capture images
    if not ok:
        continue
    h, w, _ = img.shape  # get the dimensions of our image

    empty_img = 255 * np.ones((h, w, 3), np.uint8)  # create an empty white image with the size of our frame

    img = cv2.flip(img, 1)  # the frame is mirrored so we flip it
    for color in Color_Circle:  # display the color palette
        # print(color)
        cv2.circle(img, Color_Circle[color]["Center"],
                   Color_Circle[color]["Radius"],
                   Color_Circle[color]["Color"], -1)
        cv2.circle(empty_img, Color_Circle[color]["Center"],
                   Color_Circle[color]["Radius"],
                   Color_Circle[color]["Color"], -1)

    RGB_img = cv2.cvtColor(img,
                           cv2.COLOR_BGR2RGB)  # convert the frame from BGR to RGB in order to process it correctly with mediapipe
    results = hands.process(
        RGB_img)  # launch the detection and tracking process on our img and store the results in "results"

    if results.multi_hand_landmarks:  # if a hand is detected
        for handlm in results.multi_hand_landmarks:
            for id, lm in enumerate(handlm.landmark):
                # print(handlm.landmark)
                lm_pos = (int(lm.x * w), int(lm.y * h))  # get landmarks positions
                mp_draw.draw_landmarks(img, handlm, mp_hands.HAND_CONNECTIONS)  # draw the landmarks
                if (id % 4 == 0):  # if a landmark is a fingertip ( 0,4,8,12,16,20)
                    tips_pts = np.append(tips_pts, lm_pos)  # append fingertips coordinates to tips_pts array
                    tips_pts = tips_pts.reshape((-1, 1, 2))
                    # print(len(tips_pts))

                    while (len(tips_pts) >= 5):  # keep array length constant = 5
                        tips_pts = np.delete(tips_pts, len(tips_pts) - 5, 0)
                if id == 8:  # if we detect the index finger tip
                    cv2.circle(img, lm_pos, 6, (255, 255, 255), -1)
                    for color in Color_Circle:  # calculate the distance between the index finger tip and each color in tha palette
                        Color_Circle[color]["Distance"] = distance(lm_pos, Color_Circle[color]["Center"])
                        cv2.line(img, lm_pos, Color_Circle[color]["Center"], Color_Circle[color]["Color"], 3)
                        cv2.putText(img, str(Color_Circle[color]["Distance"]), Color_Circle[color]["Center"],
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
                        if Color_Circle[color][
                            "Distance"] < 35:  # if the index is close enough to a color then this color becomes selected or "active"
                            for i in Color_Circle:
                                Color_Circle[i]["is Active"] = False  # deactivate unselected colors

                            Color_Circle[color]["is Active"] = True

                        if Color_Circle[color]["is Active"] == True:
                            cv2.circle(empty_img, lm_pos, 8, Color_Circle[color]["Color"], -1)

                            if (Is_in_Draw_Position(handlm.landmark, w, h)):  # if we are in draw position
                                # print(Is_in_Draw_Position(handlm.landmark, w, h))
                                is_Draw_curr_Frame = True  # if we are currently drawing
                                # print(" is_Draw_curr_Frame", is_Draw_curr_Frame, "is_Draw_prev_Frame",
                                #       is_Draw_prev_Frame)
                                if (is_Draw_prev_Frame == False) and (
                                        is_Draw_curr_Frame == True):  # if we just started a drawing sequence
                                    Color_Circle[color]["Drawing"].append(
                                        np.array([[]], np.int32))  # append drawing coordinates in a numpy array

                                Color_Circle[color]["Drawing"][len(Color_Circle[color]["Drawing"]) - 1] = \
                                    np.append(
                                        Color_Circle[color]["Drawing"][len(Color_Circle[color]["Drawing"]) - 1],
                                        lm_pos)
                                # print(Color_Circle[color]["Drawing"])

                                Color_Circle[color]["Drawing"][len(Color_Circle[color]["Drawing"]) - 1] = \
                                    Color_Circle[color]["Drawing"][len(Color_Circle[color]["Drawing"]) - 1].reshape(
                                        (-1, 1, 2))

                            else:
                                # print(Is_in_Draw_Position(handlm.landmark, w, h))
                                is_Draw_curr_Frame = False

                            is_Draw_prev_Frame = is_Draw_curr_Frame
                            # print(" *** is_Draw_curr_Frame", is_Draw_curr_Frame, "is_Draw_prev_Frame",
                            #       is_Draw_prev_Frame)

                            print(len(Color_Circle[color]["Drawing"]))

                Box_corner1, Box_corner2 = Bounding_box_coords(handlm.landmark)

                cv2.rectangle(img, Box_corner1, Box_corner2, (0, 0, 255), 2)  # draw a bounding box around the hand
                # print(Box_corner2 , h , w)
                # cv2.circle(img,Box_center,1 ,(255,0,0),2)
                cv2.polylines(img, [tips_pts], False, (255, 0, 255), 2)  # draw a polygone around the hand

    for color in Color_Circle:  # display our drawing
        for i in range(0, len(Color_Circle[color]["Drawing"])):
            cv2.polylines(empty_img, [Color_Circle[color]["Drawing"][i]], False, Color_Circle[color]["Color"], 5)

    curr_frame_time = time.time()
    delta_time = curr_frame_time - prev_frame_time
    fps = int(1 / delta_time)
    prev_frame_time = curr_frame_time
    cv2.putText(img, "FPS : " + str(fps), (int(0.01 * w), int(0.2 * h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
    cv2.imshow("final img", img)
    cv2.imshow("empty img", empty_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # "q" to quit
        break
    elif cv2.waitKey(1) & 0xFF == ord("c"):  # "c" to clear drawing
        for color in Color_Circle:
            Color_Circle[color]["Drawing"].clear()
            pass

cap.release()