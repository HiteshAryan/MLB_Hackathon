import numpy as np
import cv2

class ViewTransformerBat():
    def __init__(self):
        pitch_length = 18.4

        self.pixel_vertices = np.array([
            (643, 119),  # Top-left
            (643, 395),  # Bottom-left
            (877, 119),  # Top-right
            (877, 395)
        ])

        self.target_vertices = np.array([
            (0, 0),  # Top-left
            (0, 2.5),  # Bottom-left
            (2.5, 0),  # Top-right
            (2.5, 2.5)
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
            for frame_num, track in enumerate(object_tracks):
                for _, track_info in track.items():
                    position = track_info['position']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][1]['position_transformed'] = position_transformed