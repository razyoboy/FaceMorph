import os.path

import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
from face_landmark_detection import facial_landmarks_detector
from face_morph import morph_triangle

def main(predictor_path, filename1, filename2, alpha, include_borders, export_image):
    # Read images
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # Read array of corresponding points
    points1 = np.array(facial_landmarks_detector(predictor_path, filename1, include_borders))
    points2 = np.array(facial_landmarks_detector(predictor_path, filename2, include_borders))
    points = (1 - alpha) * points1 + alpha * points2

    # Allocate space for final output
    img_morph = np.zeros(img1.shape, dtype = img1.dtype)

    tri = Delaunay(points)

    # Read triangles from tri.txt
    for simplex in tri.simplices:
        x,y,z = simplex

        x = int(x)
        y = int(y)
        z = int(z)

        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [ points[x], points[y], points[z] ]

        # Morph one triangle at a time.
        morph_triangle(img1, img2, img_morph, t1, t2, t, alpha)

    # Display Result
    plt.imshow(cv2.cvtColor(np.uint8(img_morph), cv2.COLOR_BGR2RGB))
    plt.axis('off')  # No axes for this plot

    if export_image:
        filename1_without_ext = os.path.splitext(os.path.basename(filename1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(filename2))[0]
        output_path = f'output/morphed_{filename1_without_ext}_{filename2_without_ext}_alpha_{alpha}.jpg'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    plt.show()

if __name__ == '__main__' :
    predictor_path = 'model/shape_predictor_68_face_landmarks.dat'
    filename1 = 'img/donald_trump.jpg'
    filename2 = 'img/hillary_clinton.jpg'
    alpha = 0.5
    include_borders = True
    export_image = False

    main(predictor_path, filename1, filename2, alpha, include_borders, export_image)
