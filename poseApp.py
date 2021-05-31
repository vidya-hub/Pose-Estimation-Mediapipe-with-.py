import mediapipe as mp
from cv2 import cv2
mp_drawing = mp.solutions.drawing_utils
mp_holis = mp.solutions.holistic
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_holistic = mp.solutions.holistic.Holistic()

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_holistic.process(imgRgb)
    annotated_image = img.copy()
    if results.pose_landmarks and results.face_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_holis.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image, results.face_landmarks, mp_holis.FACE_CONNECTIONS)
    cv2.imshow("Image", annotated_image)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()
