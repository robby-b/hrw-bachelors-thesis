import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction.input_representation.static_layers import get_lanes_in_radius
from nuscenes.prediction.helper import convert_global_coords_to_local

#########################################################################################################################################################

# polylines are saved in formate [x_start, y_start, x_end, y_end, object_type, timestep, id]
class Polylines:
    def __init__(self, maps, helper, input_len, pred_len):

        self.sample_freq = 2 # nuscenes is sampled with 2Hz
        self.input_len = input_len * self.sample_freq
        self.pred_len = pred_len * self.sample_freq
        self.helper = helper
        self.maps = maps
        self.ego_range = (40, 40, 50, 100)

    def get_past_future_agent(self, instance_token, sample_token, seconds, direction):

        # init 
        time_elapsed = 0                    
        annotations = []

        # start init
        starting_annotation = next_annotation = self.helper.data.get('sample_annotation', self.helper.inst_sample_to_ann[(sample_token, instance_token)])
        starting_time = self.helper.data.get('sample', starting_annotation['sample_token'])['timestamp'] / 1e6

        # append start annotation when looking into the past
        if direction == 'prev':
            annotations.append(starting_annotation)
            seconds -= 1/self.sample_freq

        # get annotations
        while abs(time_elapsed) <= (seconds + 0.15) and next_annotation[direction] != '':
            
            # get next annotation and timestep
            next_annotation = self.helper.data.get('sample_annotation', next_annotation[direction])
            time_elapsed = self.helper.data.get('sample', next_annotation['sample_token'])['timestamp'] / 1e6 - starting_time
            
            # if in timeframe append data
            if abs(time_elapsed) < (seconds + 0.15): annotations.append(next_annotation)

        # get coordinates from annotations
        coords = np.array([r['translation'][:2] for r in annotations])

        # flip array for past coordinates
        if direction == 'prev': coords = coords[::-1]

        # rotate coordinates to ego frame
        coords = np.array([convert_global_coords_to_local(pos, self.annotation['translation'], self.annotation['rotation']).squeeze() for pos in coords])

        return coords

    def process_ego_agent(self):

        # get coordinates
        x_ego = self.get_past_future_agent(self.instance_token, self.sample_token, seconds=self.input_len//self.sample_freq, direction='prev')
        y_ego = self.get_past_future_agent(self.instance_token, self.sample_token, seconds=self.pred_len//self.sample_freq, direction='next')
        
        # append start coordinates to future
        y_ego = np.concatenate([[[0, 0]], y_ego])

        # pad data
        if len(x_ego) < self.input_len:     x_ego = np.pad(x_ego, ((self.input_len - len(x_ego), 0), (0, 0)))
        if len(y_ego) < self.pred_len+1:    y_ego = np.pad(y_ego, ((0, (self.pred_len+1) - len(y_ego)), (0, 0)))

        # shape to format [x_start, y_start, x_end, y_end] 
        x_ego = sliding_window_view(x_ego, (2, 2)).reshape(-1, 4)
        y_ego = sliding_window_view(y_ego, (2, 2)).reshape(-1, 4)

        # fix last value of sliding window for padded coordinates
        x_ego = np.array([[0,0,0,0] if (vec[0] == vec[1] == 0) else vec for vec in x_ego]) # error for the beginning of window
        y_ego = np.array([[0,0,0,0] if (vec[2] == vec[3] == 0) else vec for vec in y_ego]) # error for the end of window

        return x_ego, y_ego

    def process_agents(self):

        # get coordinates and annotations for all agents
        sample_record = self.helper.data.get('sample', self.sample_token)
        past_samples = {}
        for annotation in sample_record['anns']:
            annotation_record = self.helper.data.get('sample_annotation', annotation)
            if annotation_record['instance_token'] != self.annotation['instance_token']: # append to dict if not ego agent
                sequence = self.get_past_future_agent(annotation_record['instance_token'], annotation_record['sample_token'], seconds=self.input_len//self.sample_freq, direction='prev')
                if len(sequence) > 1: # dont add empty entrys
                    past_samples[annotation_record['instance_token']] = sequence

        # go through samples and get agents
        agent_types = []
        agents = []
        for key, value in past_samples.items():
            
            # get agent type
            agent_type = self.helper.get_sample_annotation(key, self.sample_token)['category_name']

            # add raodpoint if it is in scene boundarys
            if (-self.ego_range[0] <= value[0, 0] <= self.ego_range[1]) and (-self.ego_range[2] <= value[0, 1] <= self.ego_range[3]):

                # only vehicles and people are considered according to prediction challenge
                if "vehicle" in agent_type or "human" in agent_type:

                    # get current agent
                    curr_agent = past_samples[key][::-1]

                    # pad if agent is not observed over entire input horizon
                    if len(curr_agent) < self.input_len:    curr_agent = np.pad(curr_agent, ((self.input_len - len(curr_agent), 0), (0, 0)))

                    # append agent and type to agent list
                    agents.append(curr_agent)
                    agent_types.append(agent_type)

        # shape agents to format [x_start, y_start, x_end, y_end]
        agents = [sliding_window_view(agent, (2, 2)).reshape(-1, 4) for agent in agents]

        # fix last value of sliding window
        x_agents = []
        for agent in agents:
            x_agents.append([[0,0,0,0] if (vec[0] == vec[1] == 0) else vec for vec in agent])

        return np.array(x_agents)

    def process_map(self):

        # get lanes
        lanes = get_lanes_in_radius(x=self.annotation["translation"][0], y=self.annotation["translation"][1], radius=200, map_api=self.nusc_map, discretization_meters=2.0)

        # consturct map array
        x_map = []
        for _, value in lanes.items():

            road = []
            for items in value:
                
                # get coordinates and rotate them to corrent scene frame 
                x_coord, y_coord = items[:2][0], items[:2][1]
                rotated_pt = convert_global_coords_to_local([x_coord, y_coord], self.annotation['translation'], self.annotation['rotation'])

                # add raodpoint if it is in scene boundarys
                if (-self.ego_range[0] <= rotated_pt[0, 0] <= self.ego_range[1]) and (-self.ego_range[2] <= rotated_pt[0, 1] <= self.ego_range[3]): road.append(np.squeeze(rotated_pt))

            # if road had elements add road to map
            if len(road) > 0: x_map.append(road)

        # shape agents to format [x_start, y_start, x_end, y_end] and norm to last observed ego position
        x_map = [sliding_window_view(lane_element, (2, 2)).reshape(-1, 4) for lane_element in x_map if len(lane_element) > 1]

        return x_map
    
    def add_timesteps(self, x, x_map):

        # ad timesteps to agents and 0 timestep to map data
        x = [[np.append(vec, t+1) for t, vec in enumerate(element)] for element in x]
        x_map = [[np.append(vec, 0) for vec in element] for element in x_map]

        return x, x_map

    def add_type(self, x, x_map):

        # ad type to agents and map (this is a dummy for now and only adds 1 for agent and 2 for lanes)
        x = [[np.append(vec, 1) if np.all(vec != [0,0,0,0]) else np.append(vec, 0) for vec in element] for element in x] # padded agents get type 0
        x_map = [[np.append(vec, 2) for vec in element] for element in x_map]

        return x, x_map

    def add_poylline_id(self, x, x_map):

        # add polyline id to agents and map
        x = [[np.append(vec, i) for vec in element] for i, element in enumerate(x)]

        # add polyline id to map lanes
        last_id = x[-1][-1][-1] + 1
        x_map = [[np.append(vec, i+last_id) for vec in element] for i, element in enumerate(x_map)]

        return x, x_map

    def remove_dims(self, x, x_map):

        polylines_traj = []
        for agent in x:
            for v in agent:
                polylines_traj.append(v.tolist())

        polylines_lane = []
        for lane in x_map:
            for v in lane:
                polylines_lane.append(v.tolist())

        return polylines_traj, polylines_lane

    def create(self, scene):

        # get token and anotation for current data
        self.instance_token, self.sample_token = scene.split("_")
        self.annotation = self.helper.get_sample_annotation(self.instance_token, self.sample_token)

        # get current map
        map_name = self.helper.get_map_name_from_sample_token(self.sample_token)
        self.nusc_map = self.maps[map_name]

        # process agent and map data
        x_ego, y_ego = self.process_ego_agent()
        x_agent = self.process_agents()
        x_map = self.process_map()

        # concat agents (but only if other agents are present)
        if len(x_agent) > 0:    x = np.concatenate((x_ego[np.newaxis], x_agent))
        else:                   x = x_ego[np.newaxis]

        # format and add info to coordinates
        x, x_map = self.add_type(x, x_map)
        x, x_map = self.add_timesteps(x, x_map)
        x, x_map = self.add_poylline_id(x, x_map)
        x, x_map = self.remove_dims(x, x_map)

        # some scenes dont have map data -> pad dummy lanes
        if len(x_map) < 1:  x_map = np.zeros((2, 7))

        return np.array(x), np.array(x_map), np.array(y_ego)

#########################################################################################################################################################

def get_scenes(data_dir, dataset, left_handed):
    
    # data folder contains full dataset
    if 'v1.0-trainval' in os.listdir(data_dir): 
        nuscenes = NuScenes('v1.0-trainval', dataroot=data_dir, verbose=0)
        raw_scenes = get_prediction_challenge_split(dataset[:-1], dataroot=data_dir)
    # data folder only contains mini dataset
    else:   
        nuscenes = NuScenes('v1.0-mini', dataroot=data_dir, verbose=0) 
        raw_scenes = get_prediction_challenge_split('mini_'+dataset[:-1], dataroot=data_dir)

    # get helper
    helper = PredictHelper(nuscenes)

    # filter scenes according to trafic rule (singapore data is left handed)
    scenes = []
    for scene in raw_scenes:

            # get all ego positions
            _, sample_token = scene.split("_")
            map_name = helper.get_map_name_from_sample_token(sample_token)

            # save data of correct length and trafic rule
            if ('singapore' in map_name and left_handed) or ('boston' in map_name and not left_handed):
                scenes.append(scene) 

    return scenes


def get_polyline_class(data_dir, input_len, pred_len):
    
    # data folder contains full dataset
    if 'v1.0-trainval' in os.listdir(data_dir): nuscenes = NuScenes('v1.0-trainval', dataroot=data_dir, verbose=0)
    # data folder only contains mini dataset
    else:                                       nuscenes = NuScenes('v1.0-mini', dataroot=data_dir, verbose=0) 

    # get helper
    helper = PredictHelper(nuscenes)

    # get map
    json_files = filter(lambda f: "json" in f and "prediction_scenes" not in f, os.listdir(os.path.join(data_dir, "maps", "expansion")))
    maps = {}
    for map_file in json_files:
        map_name = str(map_file.split(".")[0])
        maps[map_name] = NuScenesMap(data_dir, map_name=map_name)

    # create and return polyline class
    return  Polylines(maps, helper, input_len, pred_len)
    