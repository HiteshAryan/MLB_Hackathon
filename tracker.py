import cv2
from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_radius_of_circle

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 30
        detections = []
        for i in range(0, len(frames), batch_size):
                detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.4)
                detections += detections_batch

        return detections

    def add_position_to_tracks(self, tracks):
        for object, objects_tracks in tracks.items():
            for frame_num, track in enumerate(objects_tracks):
                for _, track_info in track.items():
                    if object == "baseball":
                        position = get_center_of_bbox(track_info["bbox"])
                    tracks[object][frame_num][1]["position"] = position

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                stub = pickle.load(f)
            return stub

        detections = self.detect_frames(frames)

        tracks = {
            "baseball": [],
            "batter": [],
            "pitcher": []
        }

        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            print("cls_name", cls_name)
            cls_names_inv = {v:k for k,v in cls_name.items()}

            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            # print("detection supervision: ", detection_supervision)

            # print(type(detection_supervision))
            # detection_supervision.xyxy = sv.pad_boxes(detection_supervision.xyxy, px=10, py=10)

            # Track Objects
            # detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            # print("detection_with_tracks: ", detection_with_tracks)

            #print(detection_with_tracks)

            tracks["baseball"].append({})
            tracks["batter"].append({})
            tracks["pitcher"].append({})

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["batter"]:
                    tracks["batter"][frame_num][1] = {"bbox": bbox}

                if cls_id == cls_names_inv["pitcher"]:
                    tracks["pitcher"][frame_num][1] = {"bbox": bbox}

                if cls_id == cls_names_inv["baseball"]:
                    tracks["baseball"][frame_num][1] = {"bbox": bbox}


        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        # interpolate between first and last values
        # first_valid_index = df_ball_positions.first_valid_index()
        # last_valid_index = df_ball_positions.last_valid_index()

        df_ball_positions = df_ball_positions.interpolate(limit = 2, limit_direction = 'forward')
        # df_ball_positions.loc[first_valid_index:last_valid_index] = df_ball_positions.loc[first_valid_index:last_valid_index].interpolate()
        # df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {} if all(np.isnan(val) for val in x) else {1: {'bbox': x}}
            for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

    def draw_circle(self, frame, bbox, color):
        radius = get_radius_of_circle(bbox)
        center = get_center_of_bbox(bbox)
        cv2.circle(frame, center, 10, color, lineType=cv2.LINE_AA, thickness = 3)
        #cv2.circle(frame, center, radius-7, (0, 175, 175), lineType=cv2.FILLED, thickness = -1)

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            pitcher_dict = tracks["pitcher"][frame_num]
            baseball_dict = tracks["baseball"][frame_num]
            batter_dict = tracks["batter"][frame_num]
            bat_dict = tracks["bat"][frame_num]

            # Draw ball
            for _, ball in baseball_dict.items():
                frame = self.draw_circle(frame, ball["bbox"], (153, 51, 220))

            # Draw bat
            for _, bat in bat_dict.items():
                frame = self.draw_circle(frame, bat["bbox"], (153, 255, 204))
            output_video_frames.append(frame)

        return output_video_frames