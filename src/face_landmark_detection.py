import os
import dlib
import numpy as np


def facial_landmarks_detector(predictor_path: str, picture: str, include_borders: bool):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    print("Processing file: {}".format(picture))
    img = dlib.load_rgb_image(picture)

    height, width, _ = img.shape

    detectors = detector(img, 1)
    print("Number of faces detected: {}".format(len(detectors)))
    for k, d in enumerate(detectors):
        shape = predictor(img, d)
        points = np.zeros((68, 2), dtype=np.int32)

        for i in range(0, 68):
            points[i] = (shape.part(i).x, shape.part(i).y)

        # Adding border points if include_border is true
        if include_borders:
            border_points = [
                (0, 0),
                (width // 2, 0),
                (width - 1, 0),
                (width - 1, height // 2),
                (width - 1, height - 1),
                (width // 2, height - 1),
                (0, height - 1),
                (0, height // 2)
            ]
            points = np.vstack([points, border_points])

        return points


if __name__ == "__main__":

    faces = [
        'img/donald_trump.jpg',
        'img/hillary_clinton.jpg'
    ]

    predictor_path = 'model/shape_predictor_68_face_landmarks.dat'
    res = facial_landmarks_detector(predictor_path, faces[1], True)

    from scipy.spatial import Delaunay
    tri = Delaunay(res)

    import matplotlib.pyplot as plt
    plt.triplot(res[:,0], res[:,1], tri.simplices)
    plt.plot(res[:,0], res[:,1], 'o')
    plt.show()
