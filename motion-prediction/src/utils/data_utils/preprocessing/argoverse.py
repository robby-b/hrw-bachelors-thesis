import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

from src.utils.data_utils.preprocessing.helper import rot_2D, get_angle

#########################################################################################################################################################

# polylines are saved in formate [x_start, y_start, x_end, y_end, object_type, timestep, id]
class Polylines:
    def __init__(self, data_dir, input_len, pred_len, left_handed=False):

        # meta data
        self.hz = 10
        self.input_len = input_len * self.hz 
        self.pred_len = pred_len * self.hz 
        self.observ_len = self.input_len+self.pred_len
        self.left_handed = left_handed
        self.ego_range = (40, 40, 50, 100)

        # get map and loader
        self.map = ArgoverseMap()
        self.loader = ArgoverseForecastingLoader(data_dir+'data/')


    def get_agent_polylines(self):
        
        # go through all agents and create polylines
        for _, group_data in self.scene.groupby("TRACK_ID"):

            # remove rows with recurring timestamps
            group_data = group_data[~group_data.duplicated(subset='TIMESTAMP', keep='first')]
            
            # get agent info
            x = group_data["X"].values
            y = group_data["Y"].values
            timesteps = group_data["TIMESTAMP"].values
            
            # only create polylines if there is more than one value
            if len(x) > 1 and group_data["OBJECT_TYPE"].values[0] != "AGENT":

                # set id
                agent_id = self.polyline_id
                self.polyline_id += 1

                # create agent polylines
                c = self.transform_coordinates(x, y)
                agent = np.column_stack((c[:,0], c[:,1], c[:,2], c[:,3], np.ones(len(c)), timesteps[1:], np.full(len(c), agent_id)))

                # pad agent
                agent = self.pad_agent(agent, agent_id)
                agent = agent[:self.input_len-1] # only take past trajectories as input data

                # append vectors of padded agent
                for polyline in agent:
                    if  (-self.ego_range[0] < polyline[0] < self.ego_range[1]) and (-self.ego_range[2] < polyline[1] < self.ego_range[3]) and \
                        (-self.ego_range[0] < polyline[2] < self.ego_range[1]) and (-self.ego_range[2] < polyline[3] < self.ego_range[3]):
                        self.polylines_traj.append(polyline)
                    else:
                        self.polylines_traj.append(np.concatenate(([0, 0, 0, 0, 0], polyline[5:])))

        
    def get_lane_polylines(self):

        # Get map lanes which lie within the range of trajectories
        for _, lane_properties in self.map.city_lane_centerlines_dict[self.scene["CITY_NAME"].values[0]].items():

            # get lane info
            intersec = lane_properties.is_intersection
            traffic_controll = lane_properties.has_traffic_control
            lane = lane_properties.centerline

            # get lane coordinates
            if np.min(lane[:, 0]) < self.x_max and np.min(lane[:, 1]) < self.y_max and np.max(lane[:, 0]) > self.x_min and np.max(lane[:, 1]) > self.y_min:   

                # get lane type
                x = lane[:,0]
                y = lane[:,1]
                lane_type = int(intersec)*2 + int(traffic_controll) + 2
                lane_elements = len(x)

                c = self.transform_coordinates(x, y)
                lane = np.column_stack((c[:,0], c[:,1], c[:,2], c[:,3], np.full(lane_elements-1, lane_type), np.zeros(lane_elements-1), np.full(lane_elements-1, self.polyline_id)))
                self.polyline_id += 1

                # append polylines of lane
                for polyline in lane:
                    if  (-self.ego_range[0] < polyline[0] < self.ego_range[1]) and (-self.ego_range[2] < polyline[1] < self.ego_range[3]) and \
                        (-self.ego_range[0] < polyline[2] < self.ego_range[1]) and (-self.ego_range[2] < polyline[3] < self.ego_range[3]):
                        self.polylines_lane.append(polyline)
        
        # pad zeros if no map data is available
        if len(self.polylines_lane) == 0: self.polylines_lane.append(np.zeros(7))


    def get_ego_agent(self):

        for _, group_data in self.scene.groupby("TRACK_ID"):
            if group_data["OBJECT_TYPE"].values[0] == "AGENT":
                
                # remove rows with recurring timestamps
                group_data = group_data[~group_data.duplicated(subset='TIMESTAMP', keep='first')]

                # get agent info
                x = group_data["X"].values
                y = group_data["Y"].values
                timesteps = group_data["TIMESTAMP"].values

                # get last positions and angle
                c = np.zeros((self.observ_len, 2))
                c[timesteps.astype(int), :] = np.column_stack((x, y)) # pad missing coordinates

                t_back = 1
                self.ego_last_pos = c[self.input_len-1] # get last position
                while np.all(self.ego_last_pos == [0, 0]): 
                    self.ego_last_pos = c[self.input_len-1-t_back]
                    t_back += 1

                t_back += 1
                ego_second_last_pos = c[self.input_len-t_back] # get scond last postition
                while np.all(ego_second_last_pos == self.ego_last_pos): 
                    ego_second_last_pos = c[self.input_len-t_back-1]
                    t_back += 1
                self.ego_angle = get_angle(ego_second_last_pos, self.ego_last_pos)  # get last angle of agent

                # create agent polylines
                c = self.transform_coordinates(x, y)
                ego_agent = np.column_stack((c[:,0], c[:,1], c[:,2], c[:,3], np.ones(len(c)), timesteps[1:], np.zeros(len(c))))

                # pad missing data
                ego_agent = self.pad_agent(ego_agent, 0)

                # append vectors of padded agent
                for polyline in ego_agent[:self.input_len-1]:
                    self.polylines_traj.append(polyline)

                # get ground truth for scene
                self.future_traj = ego_agent[self.input_len-1:]


    def transform_coordinates(self, x, y):

        c = np.column_stack((x, y))-self.ego_last_pos
        c = np.apply_along_axis(rot_2D, 1, c, self.ego_angle, self.left_handed)
        return sliding_window_view(c, (2, 2)).reshape(-1, 4)
    

    def pad_agent(self, agent, agent_id):

        agent_padded = np.zeros((self.observ_len-1, 7))
        agent_padded[:,-2] = np.arange(1, self.observ_len)  # timestep vector
        agent_padded[:,-1] = np.full(self.observ_len-1, agent_id) # id vector
        row_indices = agent[:, -2].astype(int)-1
        agent_padded[row_indices, :] = agent
        return agent_padded


    def create(self, scene):

        # load scene
        self.scene = self.loader.get(scene).seq_df

        # get min max values
        self.x_min, self.x_max = min(self.scene["X"]), max(self.scene["X"])
        self.y_min, self.y_max = min(self.scene["Y"]), max(self.scene["Y"])
        self.t_min = min(self.scene["TIMESTAMP"])

        # correct datafram
        self.scene["TIMESTAMP"] = self.scene["TIMESTAMP"].apply(lambda t: np.round((t-self.t_min) * self.hz, 0)) # correct timesteps 
        self.scene = self.scene[self.scene["TIMESTAMP"] < self.observ_len] # filter timesteps that are out of observation horizon
        
        # init polylines and id
        self.polyline_id = 1
        self.polylines_traj = []
        self.polylines_lane = []

        # process data and get polylines
        self.get_ego_agent()
        self.get_agent_polylines()
        self.get_lane_polylines()

        # return np arrays
        return np.array(self.polylines_traj, float), np.array(self.polylines_lane, float), np.array(self.future_traj, float)


#########################################################################################################################################################

def get_scenes(data_dir):

    # return filtered paths
    return [data_dir+'data/'+file_name for file_name in os.listdir(data_dir+'data/')]


def get_polyline_class(data_dir, input_len, pred_len, left_handed):

    # create and return polyline class
    return  Polylines(data_dir, input_len, pred_len, left_handed)
