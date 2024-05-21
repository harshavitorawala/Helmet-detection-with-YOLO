from ultralytics import YOLO
import cv2

model=YOLO('best.pt')


video_path = '<VIDEO_PATH>'
cap = cv2.VideoCapture(video_path)

ret = True
while ret:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(1080,500))
    
    if ret:

        results = model.track(frame, persist=True)
        frame_ = results[0].plot()

        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
