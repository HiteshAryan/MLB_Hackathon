import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        pitch_length = 18.4

        self.pixel_vertices = np.array([
            (453, 584), (680, 427), (489, 267), (682, 255)
        ])

        self.target_vertices = np.array([
            (0, 18.44), (18.44, 18.44), (0, 0), (18.44, 0)
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        # is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        # if not is_inside:
        #     return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)

        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            if object == "pitcher" or object == "batter":
                continue
            for frame_num, track in enumerate(object_tracks):
                for _, track_info in track.items():
                    position = track_info['position']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][1]['position_transformed'] = position_transformed