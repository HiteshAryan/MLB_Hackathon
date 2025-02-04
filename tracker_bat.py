import cv2
from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../')
from utils import *

class TrackerBat:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
                detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.3)
                detections += detections_batch

        return detections

    def add_position_to_tracks(self, tracks):
        for object, objects_tracks in tracks.items():
            for frame_num, track in enumerate(objects_tracks):
                for _, track_info in track.items():
                    if object == "bat":
                        position = get_center_of_bbox(track_info["bbox"])
                    tracks[object][frame_num][1]["position"] = position

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                stub = pickle.load(f)
            return stub

        detections = self.detect_frames(frames)

        tracks = {"bat": []}

        for frame_num, detection in enumerate(detections):
            cls_name = {k: ('bat' if v == '0' else v) for k, v in detection.names.items() if v == '0'}
            cls_names_inv = {v:k for k,v in cls_name.items()}

            # Convert to supervision detection format
            #detection.class_names = np.array([detection.names[int(i)] for i in detection.names])
            detection_supervision = sv.Detections.from_ultralytics(detection)
            # print("detection supervision: ", detection_supervision)

            #print(type(detection_supervision))
            #print(detection_supervision.data)
            # detection_supervision.xyxy = sv.pad_boxes(detection_supervision.xyxy, px=10, py=10)

            # Track Objects
            # detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            # print("detection_with_tracks: ", detection_with_tracks)

            tracks["bat"].append({})

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["bat"]:
                    tracks["bat"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def interpolate_bat_positions(self, bat_positions):
        bat_positions = [x.get(1, {}).get('bbox', []) for x in bat_positions]
        df_bat_positions = pd.DataFrame(bat_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        # interpolate between first and last values
        # first_valid_index = df_ball_positions.first_valid_index()
        # last_valid_index = df_ball_positions.last_valid_index()

        df_bat_positions = df_bat_positions.interpolate(limit = 3, limit_direction = 'both')
        # df_ball_positions.loc[first_valid_index:last_valid_index] = df_ball_positions.loc[first_valid_index:last_valid_index].interpolate()
        # df_ball_positions = df_ball_positions.bfill()

        bat_positions = [
            {} if all(np.isnan(val) for val in x) else {1: {'bbox': x}}
            for x in df_bat_positions.to_numpy().tolist()
        ]

        return bat_positions

    def draw_circle(self, frame, bbox, color, batter_center, frame_num, freeze_index, shadow = 0, object = "ball"):
        radius = get_radius_of_circle(bbox)
        center = get_center_of_bbox(bbox)
        top_right_med_corner = get_right_med_corner(bbox, center)
        bottom_left_med_corner = get_left_med_corner(bbox, center)
        # if frame_num > freeze_index:
        #     return frame

        # if frame_num == freeze_index:
        #     cv2.circle(frame, center, 10 - shadow, color, lineType=cv2.LINE_AA, thickness=-1)

        if "bat" in object and bbox[0] > batter_center[0]-20:
            cv2.circle(frame, top_right_med_corner, 10-shadow, color, lineType=cv2.LINE_AA, thickness = -1)
        elif "bat" in object and bbox[0] < batter_center[0]-20:
            cv2.circle(frame, bottom_left_med_corner, 10 - shadow, color, lineType=cv2.LINE_AA, thickness=-1)
        else:
            cv2.circle(frame, center, 10 - shadow, color, lineType=cv2.LINE_AA, thickness=-1)

        return frame

    def draw_annotations(self, video_frames, tracks, num_lags, batter_center, freeze_index):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            bat_dict = tracks["bat"][frame_num]
            baseball_dict = tracks["baseball"][frame_num]
            shadow_dict = {}

            shadow_color = {
                "baseball": {1: (204,102,0), 2: (255,178,102), 3: (153,255,255), 4: (154,204,255), 5: (102,178,255)},
                "bat": {1: (153,76,0), 2: (204,102,0), 3: (255,128,0), 4: (255,153,51), 5: (255,178,102)}
            }

            for i in range(num_lags):
                shadow_dict[f"ball_shadow{i + 1}"] = tracks[f"baseball_shadow{i+1}"][frame_num]
                shadow_dict[f"bat_shadow{i + 1}"] = tracks[f"bat_shadow{i + 1}"][frame_num]

            # Draw bat
            for _, bat in bat_dict.items():
                frame = self.draw_circle(frame, bat["bbox"], (102,0,204), batter_center,
                                         frame_num, freeze_index, object = "bat")

            for i in range(num_lags):
                for _, bat in shadow_dict[f"bat_shadow{i+1}"].items():
                    frame = self.draw_circle(frame, bat["bbox"], (127+10*i,30*i,255),
                                             batter_center, frame_num, freeze_index, shadow = i, object = "bat")

            # Draw ball
            for _, ball in baseball_dict.items():
                frame = self.draw_circle(frame, ball["bbox"], (153,153,0), batter_center,
                                         frame_num, freeze_index)

            for i in range(num_lags):
                for _, ball in shadow_dict[f"ball_shadow{i+1}"].items():
                    frame = self.draw_circle(frame, ball["bbox"], (255,255,20*i), batter_center,
                                             frame_num, freeze_index, shadow = i)

            output_video_frames.append(frame)

        return output_video_frames

    def shadow_freeze(self, video_frames, tracks, freeze_index, shadow_num):

        frame_data = video_frames[freeze_index]

        # Insert copies of the frame at positions n+1 to n+20
        for i in range(40):
            insert_position = freeze_index + 1 + i
            video_frames.insert(insert_position, frame_data.copy())

        for object, objects_tracks in tracks.items():

            frame_data = tracks[object][freeze_index]

            # Insert copies of the frame at positions n+1 to n+20
            for i in range(40):
                insert_position = freeze_index + 1 + i
                tracks[object].insert(insert_position, frame_data.copy())  # Copy to avoid reference issues

        for i in range(0, shadow_num):
            for idx in range(freeze_index + shadow_num + 40, freeze_index + shadow_num - i, -1):
                tracks[f"baseball_shadow{i + 1}"][idx] = {}
                tracks[f"bat_shadow{i + 1}"][idx] = {}

        return tracks

