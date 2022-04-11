import cv2
from deepface import DeepFace
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def changeStatus(key):
    global perform
    if key == ord('q'):
        perform = not perform
        if perform:
            print('Frame disappers')
        else:
            print('Frame is visibile')
cap = cv2.VideoCapture(0)
while (1):

    k = cv2.waitKey(10) & 0xFF
    changeStatus(k)

    _, frame = cap.read()
    output = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection=False)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    enforce_detection =False
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
            output['dominant_emotion'],
            (50,50),
            font, 3,
            (0,0,255),
            2,
            cv2.LINE_4);

    cv2.imshow('Frame', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()