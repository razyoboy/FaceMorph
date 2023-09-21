import os.path

import numpy as np
import cv2
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.spatial import Delaunay
from face_landmark_detection import facial_landmarks_detector
from face_morph import morph_triangle

def main(predictor_path, filename1, filename2, alpha, include_borders, export_image, display_results = True):
    # Read images
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    # Resize img2 to match img1 dimensions
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Read array of corresponding points
    points1 = np.array(facial_landmarks_detector(predictor_path, img1, include_borders))
    points2 = np.array(facial_landmarks_detector(predictor_path, img2, include_borders))
    
    # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    if points1 is None or points2 is None:
        print("Failed to detect face in one or both images.")
        return

    points = (1 - alpha) * points1 + alpha * points2

    # Allocate space for final output
    img_morph = np.zeros(img1.shape, dtype=img1.dtype)

    tri = Delaunay(points)

    # Allocate space for transparency mask
    transparency_mask = np.zeros(img1.shape[0:2], dtype=np.float32)

    # Read triangles from tri.txt
    for simplex in tri.simplices:
        x,y,z = simplex

        x = int(x)
        y = int(y)
        z = int(z)

        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morph_triangle(img1, img2, img_morph, t1, t2, t, alpha, transparency_mask)

    # Generate the 4-channel output image with transparency
    img_morph_4channel = np.zeros((img1.shape[0], img1.shape[1], 4), dtype=np.uint8)
    img_morph_4channel[..., :3] = np.uint8(img_morph)
    img_morph_4channel[..., 3] = np.uint8(transparency_mask * 255)

    if display_results:
        plt.imshow(cv2.cvtColor(np.uint8(img_morph_4channel), cv2.COLOR_BGRA2RGBA))
        plt.axis('off')  # No axes for this plot

    if export_image:
        filename1_without_ext = os.path.splitext(os.path.basename(filename1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(filename2))[0]
        if include_borders:
            output_path = f'output/morphed_{filename1_without_ext}_{filename2_without_ext}_alpha_{alpha}.png'
        else:
            output_path = f'output/morphed_{filename1_without_ext}_{filename2_without_ext}_alpha_{alpha}_mask_only.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True, format='png')

    plt.show()

if __name__ == '__main__':
    predictor_path = 'model/shape_predictor_68_face_landmarks.dat'  # Dlib's Predictor Path
    alpha = 0.5  # Alpha level (0.5 for a mix of both faces)
    
    include_borders = True  # Whether to include borders during calculations
    export_image = True  # Whether to export morphed images
    display_results = True  # Whether to display morphed images result via matplotlib
    
    BATCH_MORPH = True
    filename1 = 'img/donald_trump.jpg'
    filename2 = 'img/ted_cruz.jpg'
    folder_path = 'img/Caucasian/M'

    if BATCH_MORPH:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        for f1, f2 in combinations(files, 2):
            main(predictor_path, f1, f2, alpha, include_borders, export_image, display_results)
    else:
        main(predictor_path, filename1, filename2, alpha, include_borders, export_image)
