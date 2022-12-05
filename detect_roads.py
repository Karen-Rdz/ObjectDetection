import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image):
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [(0, 580), (0, 480), (width, 480), (width, 580)]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0) 
    canny_image = cv2.Canny(blur, 65, 110)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32),)

    lines = cv2.HoughLinesP(cropped_image, rho=3, theta=np.pi/180, threshold=160, lines=np.array([]), minLineLength=80, maxLineGap=4)

    image_with_lines = draw_lines(image, lines)
    return image_with_lines

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_images = cv2.bitwise_and(img, mask)
    return masked_images

def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

cap = cv2.VideoCapture("YOLO/original.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    frame = process_image(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):      
        break

cap.release()
cv2.destroyAllWindows()


