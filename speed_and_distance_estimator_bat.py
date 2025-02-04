import cv2
import sys
import numpy as np
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class SpeedAndDistance_EstimatorBat():
    def __init__(self):
        self.frame_window=2
        self.frame_rate=24
    
    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance= {}

        for object, object_tracks in tracks.items():

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][1]['position_transformed']
                    end_position = object_tracks[last_frame][1]['position_transformed']

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position,end_position)
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    speed_metres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_metres_per_second*3.6

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][1]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][1]['distance'] = total_distance[object][track_id]

    def contact_bat_ball(self, tracks):

        distances = []
        bat_positions = [x.get(1, {}).get('bbox', []) for x in tracks["bat"]]
        ball_positions = [x.get(1, {}).get('bbox', []) for x in tracks["baseball"]]

        for frame, (bat, ball) in enumerate(zip(bat_positions, ball_positions)):
            if not (bat and ball):
                distances.append(99999)
            else:
                bat = np.array(bat)
                ball = np.array(ball)
                distance = np.linalg.norm(bat - ball)
                distances.append(int(distance))

        return distances

    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if "shadow" in object:
                    continue

                for _, track_info in object_tracks[frame_num].items():
                   if "speed" in track_info:
                       speed = track_info.get('speed',None)
                       if object == 'baseball' or object == 'bat':
                           distance = track_info.get('distance',None)
                       # if speed is None or distance is None:
                       #     continue
                       
                       bbox = track_info['bbox']
                       position = get_center_of_bbox(bbox)
                       position = list(position)
                       position[1]+=40

                       position = tuple(map(int,position))
                       cv2.putText(frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            output_frames.append(frame)
        
        return output_frames

    def draw_speed_distance_rectangle(self, frames, tracks, freeze_index, max_ball_distance):
        output_frames=[]

        ball_speed = [x.get(1, {}).get('speed', 0) for x in tracks["baseball"]]
        bat_speed = [x.get(1, {}).get('speed', 0) for x in tracks["bat"]]
        ball_distance = [x.get(1, {}).get('distance', 0) for x in tracks["baseball"]]
        bat_distance = [x.get(1, {}).get('distance', 0) for x in tracks["bat"]]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(970,100),(1150,260),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha,0, frame)

            ball_speed_frame = ball_speed[frame_num]

            if frame_num < freeze_index:
                ball_distance_frame = ball_distance[frame_num]
            else:
                ball_distance_frame = max_ball_distance

            bat_speed_frame = bat_speed[frame_num]
            bat_distance_frame = bat_distance[frame_num]

            frame = cv2.putText(frame,f"Ball Speed : {ball_speed_frame:.2f}",(980,130), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,(0,0,0),2)
            frame = cv2.putText(frame, f"Ball Distance : {ball_distance_frame:.2f}", (980, 160), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 2)

            frame = cv2.putText(frame,f"Bat Speed: {bat_speed_frame:.2f}",(980,210), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,(0,0,0),2)
            frame = cv2.putText(frame,f"Bat Distance: {bat_distance_frame:.2f}",(980,240), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,(0,0,0),2)


            output_frames.append(frame)

        return output_frames