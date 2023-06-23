import numpy as np


# function that return the x,y coordinates of a point rotated according to the rotation matrix that is passed
def rot_2D(point, angle, mirror=False):

    # get x and y value
    px, py = point

    # get rotation angle
    rot_angle = -angle + np.pi/2

    # rotate coordinates
    qx = np.cos(rot_angle) * px - np.sin(rot_angle) * py
    qy = np.sin(rot_angle) * px + np.cos(rot_angle) * py

    # mirror coordinates
    if mirror: 
        px, py = qx, -qy
        qx = np.cos(np.pi) * px - np.sin(np.pi) * py
        qy = np.sin(np.pi) * px + np.cos(np.pi) * py

    return qx, qy


# get the rotation angle according to last observed vector
def get_angle(start_coordinate, end_coordinate):

    # get differences of coordinates
    delta_x = end_coordinate[0]-start_coordinate[0]
    delta_y = end_coordinate[1]-start_coordinate[1]

    # determine angle based on quadrant
    angle = np.arctan2(delta_y, delta_x)
    if angle<0: angle = 2*np.pi - np.abs(angle)

    return angle
