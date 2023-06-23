import torch
import numpy as np
import math
from numpy.lib.stride_tricks import sliding_window_view

# formats vector [x1,x2, ..., y1, y2, ...] back to [[x1, y1,], [x2, y2], ...]
def format_to_coordinates(vector):

    # get middle of vector and split in middle and stack 
    mid = len(vector)//2
    return torch.column_stack((vector[:mid], vector[mid:])).cumsum(axis=0)


# get last velocity of ego agent in a given scene 
def get_last_vel(input_data, sample_freq):

    # get all vectors for current batch (this is only neccesary for LSTM model and doesnt change vectornet data)
    x_batch = input_data[0].reshape(input_data[0].shape[0], -1, input_data[0].shape[-1])

    # go through all scenes
    vel_batch = []
    for x in x_batch:

        # filter out vector that contains last positions (hast to be [float, float, 0, 0, ....]) 
        vec = x[    torch.logical_and(  torch.logical_or(   x[:,0] != 0,
                                                            x[:,1] != 0),
                                        torch.logical_and(  x[:,2] == 0,
                                                            x[:,3] == 0))].squeeze()
        
        
        # check if last postion was found
        if len(vec) == 0:   vel = 0
        else :              vel = torch.sqrt(vec[0]**2 + vec[1]**2) * sample_freq
        vel_batch.append(vel)

    # return tensor
    return torch.tensor(vel_batch)


# calculate the angles between all gt vectors, given an array in the format [[x_start, y_start, x_end, y_end], [x_start, y_start, x_end, y_end], ...]
def calculate_angles(traj):

    angles = []
    for i in range(1, len(traj)):

        # get vectors
        v1 = traj[i-1]
        v2 = traj[i]

        # norm vectors to origin
        v1 = np.array([v1[2] - v1[0], v1[3] - v1[1]])
        v2 = np.array([v2[2] - v2[0], v2[3] - v2[1]])

        # check if stand still and skip vectors
        if np.all(v1==[0,0]) or np.all(v2==[0,0]): continue

        # calc dot product and norm
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        # calc angle
        cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)

        # fix numeric error
        if math.isclose(cosine_angle, 1, rel_tol=1e-6, abs_tol=0.0):    cosine_angle = 1
        elif math.isclose(cosine_angle, -1, rel_tol=1e-6, abs_tol=0.0): cosine_angle = -1

        # get angle 
        angle = math.acos(cosine_angle)
        angle_degrees = np.degrees(angle)

        # append angle
        angles.append(angle_degrees)

    return angles


# classify a scene based on gt vectors
def get_direction(traj):

    # format to vector format
    traj = sliding_window_view(traj.cpu(), (2, 2)).reshape(-1, 4)

    # filter all padded entries
    mask = (traj[:,0] != 0) | (traj[:,1] != 0) | (traj[:,2] != 0) | (traj[:,3] != 0)
    traj = traj[mask]
    if len(traj) == 0: return "other" # check if all data was filtered and return other

    # get angles
    angles = calculate_angles(traj)

    # get total x displacement from start to end pos
    dx_total = traj[-1,2] - traj[0,0]

    # get y displacement for all vectors
    dy = traj[:,3] - traj[:,1]

    # Classify the trajectory 
    # stand still (no angles), not smooth enough (angle var to high) or drive backwards more than 0.5m
    if len(angles) == 0 or np.var(angles) > 500 or min(dy) < -0.5:  return "other"    
    # traj displacement is more than 5m to the right from start position
    elif dx_total > 5:                                              return "right turn"
    # traj displacement is more than 5m to the left from start position
    elif dx_total < -5:                                             return "left turn"
    # dx is between -5m and 5m -> traj is staight
    else:                                                           return "straight drive"
    