import os
import dlib
import numpy as np


def facial_landmarks_detector(predictor_path: str, picture: str):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    print("Processing file: {}".format(picture))
    img = dlib.load_rgb_image(picture)

    detectors = detector(img, 1)
    print("Number of faces detected: {}".format(len(detectors)))
    for k, d in enumerate(detectors):
        shape = predictor(img, d)
        points = np.zeros((68, 2), dtype=np.int32)

        for i in range(0, 68):
            points[i] = (shape.part(i).x, shape.part(i).y)

        return points

if __name__ == "__main__":

    faces = [
        'img/donald_trump.jpg',
        'img/hillary_clinton.jpg'
    ]

    predictor_path = 'model/shape_predictor_68_face_landmarks.dat'
    res = facial_landmarks_detector(predictor_path, faces[0])

    from scipy.spatial import Delaunay
    tri = Delaunay(res)

    import matplotlib.pyplot as plt
    plt.triplot(res[:,0], res[:,1], tri.simplices)
    plt.plot(res[:,0], res[:,1], 'o')
    plt.show()
