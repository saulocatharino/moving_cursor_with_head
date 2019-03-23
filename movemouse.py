import cv2
import dlib
import numpy as np
from imutils import face_utils
import pyautogui
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from maat import *


fig = plt.figure()
ax = fig.gca()

face_landmark_path = 'shape_predictor_68_face_landmarks.dat'


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def eye_aspect_ratio(eye):

	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	C = dist.euclidean(eye[0], eye[3])


	ear = (A + B) / (2.0 * C)

	return ear


def draw_circle(event, xxx,yyy,flags, param):
    ax.set_ylim(0, frame.shape[0])
    ax.set_xlim(0, frame.shape[1])
    ax.scatter(xxx,frame.shape[0]-yyy, alpha=0.5, color = "black")


def main():

    tresh_head = 5 
    EYE_AR_THRESH = 0.2

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)

    while cap.isOpened():
        global frame
        ret, frame = cap.read()

        frame = cv2.flip(frame,1)
        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                for faces in face_rects:
                    shape = predictor(frame, faces)
                    shape = face_utils.shape_to_np(shape)
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0

                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)

                    reprojectdst, euler_angle = get_head_pose(shape)
                    X = euler_angle[0, 0]
                    Y = -euler_angle[1, 0]
                    if ear < EYE_AR_THRESH:
                        ax.clear()

                    if  float(X) > tresh_head:
                        pyautogui.moveRel(0,1 * int(X))

                    if  float(X) < -tresh_head:
                        pyautogui.moveRel(0,-1 * int(-X))

                    if  float(Y) > tresh_head:
                        pyautogui.moveRel(1 * int(Y) ,0)

                    if  float(Y) < -tresh_head:
                        pyautogui.moveRel(-1 * int(-Y) ,0)


            try:
                cv2.setMouseCallback('Frame',draw_circle)
            except:
                print('---**-**---')
            cv2.imshow("Frame", frame)
            plt.pause(0.001) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
