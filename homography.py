import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread, imshow
from skimage import transform
from sklearn.feature_extraction import img_to_graph

image = cv2.imread('original_gsada2.jpg')

# Specify area of interest and ares of projection
# Test
# area_of_interest = [(120, 350),
#                     (370, 350),
#                     (535, 521),
#                     (0, 521)]
# area_of_projection = [(140, 200),
#                       (460, 200),
#                       (460, 700),
#                       (140, 700)]

# Test gsada2
area_of_interest = [(120, 350),
                    (370, 350),
                    (535, 521),
                    (0, 521)]
area_of_projection = [(180, 200),
                      (500, 200),
                      (500, 700),
                      (180, 700)]

# Validation 1
area_of_interest = [(120, 350),
                    (370, 350),
                    (535, 521),
                    (0, 521)]
area_of_projection = [(180, 200),
                      (500, 200),
                      (500, 700),
                      (180, 700)]

# Obtain area of projection
def project_planes(image, src, dst): 
    new_image = image.copy() 
    projection = np.zeros_like(new_image)
    cv2.polylines(new_image, np.array([src]), True, (255, 0, 0), 5)
    cv2.imshow('Area of Interest', new_image)
    # cv2.imwrite('AreaInterest.png', new_image)
    cv2.polylines(projection, np.array([dst]), True, (255, 0, 0), 5)
    # cv2.imshow('Area of Projection', projection)

project_planes(image, area_of_interest, area_of_projection)

# Calculate homography
def project_transform(image, src, dst):
    x_dst = [val[0] for val in dst] + [dst[0][0]]
    y_dst = [val[1] for val in dst] + [dst[0][1]]

    H, _ = cv2.findHomography(np.array([src]), np.array([dst]))
    img_warp = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    cv2.polylines(img_warp, np.array([dst]), True, (255, 0, 0), 5)
    # cv2.imwrite('TransformedImage.png', img_warp)
    cv2.imshow('Area Transformed', img_warp)

project_transform(image, area_of_interest, area_of_projection)

cv2.waitKey(0)
cv2.destroyAllWindows()