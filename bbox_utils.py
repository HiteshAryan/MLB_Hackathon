def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2), int((y1 + y2)/2)

def get_radius_of_circle(bbox):
    return int(bbox[2] - bbox[0])

def measure_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_right_med_corner(bbox, center):
    x1, y1, x2, y2 = bbox
    c1, c2 = center
    return int((0.4)*c1+(0.6)*x2), int((0.4)*c2+(0.6)*y1)

def get_left_med_corner(bbox, center):
    x1, y1, x2, y2 = bbox
    c1, c2 = center
    return int((0.4)*c1+(0.6)*x1), int((0.4)*c2+(0.6)*y2)