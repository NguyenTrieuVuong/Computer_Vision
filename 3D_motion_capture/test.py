import cv2
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture('Video.mp4')

detector = PoseDetector()
posList = []
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)

    if bboxInfo:
        lmString = ''
        for lm in lmList:
            if lm[1] < len(lm) and lm[2] < len(lm) and lm[3] < len(lm):
                lmString += f'{lm[1]},{img.shape[0] - lm[2]},{lm[3]},'
            else:
                # Handle the case where lm doesn't have enough elements
                pass
        posList.append(lmString)

    print(len(posList))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        with open("AnimationFile.txt", 'w') as f:
            f.writelines(["%s\n" % item for item in posList])