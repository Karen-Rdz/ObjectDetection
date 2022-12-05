import numpy as np
import cv2

# area_of_interest = [(150, 300),
#                     (320, 300),
#                     (535, 521),
#                     (0, 521)]
area_of_interest = [(120, 350),
                    (370, 350),
                    (535, 521),
                    (0, 521)]
area_of_projection = [(180, 200),
                      (500, 200),
                      (500, 700),
                      (180, 700)]

video = cv2.VideoCapture('original_gsada2.mp4')

if (video.isOpened() == False): 
    print("Error opening video stream or file")

frame_width = int(video.get(3))
frame_height = int(video.get(4))
frame_count = 0
 
out = cv2.VideoWriter('homo_gsada2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

while(video.isOpened()):
    ret, frame = video.read()
    if ret == True:
        frame_count += 1
        if frame_count == 1:
            H, _ = cv2.findHomography(np.array([area_of_interest]), np.array([area_of_projection]))

        img_warp = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))
        # cv2.polylines(img_warp, np.array([area_of_projection]), True, (255, 0, 0), 5)
        cv2.imshow('Area Transformed', img_warp)
        out.write(img_warp)
    else: 
        break
 
video.release()
cv2.destroyAllWindows()

