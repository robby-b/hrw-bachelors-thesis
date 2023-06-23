from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneType, LaneMarkType
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import TrackCategory, ObjectType

from src.utils.data_utils.preprocessing.helper import rot_2D

#########################################################################################################################################################

# polylines are saved in formate [x_start, y_start, x_end, y_end, object_type, timestep, id]
class Polylines:
    def __init__(self, input_len, pred_len, left_handed=False):

        # meta data
        self.input_len = input_len * 10 # av2 is sampled with 10Hz
        self.pred_len = pred_len * 10   # av2 is sampled with 10Hz
        self.left_handed = left_handed
        self.pred_start_timestep = 50
        self.ego_range = (40, 40, 50, 100)

        # get dict that map object, mark and lane type to numeric values
        self.object_type = {    ObjectType.VEHICLE: 1,
                                ObjectType.PEDESTRIAN: 2,
                                ObjectType.MOTORCYCLIST: 3,
                                ObjectType.CYCLIST: 4,
                                ObjectType.BUS: 5,
                                ObjectType.STATIC: 6,
                                ObjectType.BACKGROUND: 7,
                                ObjectType.CONSTRUCTION: 8,
                                ObjectType.RIDERLESS_BICYCLE: 9,
                                ObjectType.UNKNOWN: 10}

        self.mark_type = {  LaneMarkType.DASH_SOLID_YELLOW: 1,
                            LaneMarkType.DASH_SOLID_WHITE: 2,
                            LaneMarkType.DASHED_WHITE: 3,
                            LaneMarkType.DASHED_YELLOW: 4,
                            LaneMarkType.DOUBLE_SOLID_YELLOW: 5,
                            LaneMarkType.DOUBLE_SOLID_WHITE: 6,
                            LaneMarkType.DOUBLE_DASH_YELLOW: 7,
                            LaneMarkType.DOUBLE_DASH_WHITE: 8,
                            LaneMarkType.SOLID_YELLOW: 9,
                            LaneMarkType.SOLID_WHITE: 10,
                            LaneMarkType.SOLID_DASH_WHITE: 11,
                            LaneMarkType.SOLID_DASH_YELLOW: 12,
                            LaneMarkType.SOLID_BLUE: 13,
                            LaneMarkType.NONE: 14,
                            LaneMarkType.UNKNOWN: 15}

        self.lane_type = {  LaneType.VEHICLE: 1,
                            LaneType.BIKE: 2,
                            LaneType.BUS: 3}
        
        # remember max values for types
        self.max_obj = 10
        self.max_mark = 15
        self.max_lane = 3

        # create timestep range depending the passed arguments
        self.start_time_step = self.pred_start_timestep - self.input_len
        self.end_time_step = self.pred_start_timestep + self.pred_len
        self.timesteps = range(self.start_time_step, self.end_time_step)

    def get_agent_polylines(self):

        # go through all timesteps and trajectories
        for timestep in self.timesteps:
            for track in self.scenario.tracks:
                
                # get timesteps for which actor data is valid
                actor_timesteps = np.array([state.timestep for state in track.object_states if state.timestep <= timestep])

                # check if actor has data for current timestep
                if actor_timesteps.shape[0] > 1 and actor_timesteps[-1] == timestep:

                    # get trajectory up to 5 sec mark
                    if timestep < self.pred_start_timestep:

                        # Get actor trajectory and calc start and end coordinates
                        actor_trajectory = [state.position for state in track.object_states if state.timestep == timestep or state.timestep == (timestep-1)] - self.ego_end_pos
                        x_start, y_start = rot_2D(actor_trajectory[0], self.ego_end_angle, self.left_handed)
                        x_end, y_end = rot_2D(actor_trajectory[1], self.ego_end_angle, self.left_handed)

                        # get polyline id and set to 0 if focal agent
                        if track.category != TrackCategory.FOCAL_TRACK: polyline_id = self.get_polyline_id(track.track_id)
                        else:                                           polyline_id = 0

                        # check if cordinates are in range
                        if  (-self.ego_range[0] < x_start < self.ego_range[1]) and (-self.ego_range[2] < y_start < self.ego_range[3]) and \
                            (-self.ego_range[0] < x_end < self.ego_range[1]) and (-self.ego_range[2] < y_end < self.ego_range[3]):
                            
                            # encode polyline type
                            polyline_type = self.encode_type(obj_type=self.object_type[track.object_type])

                            # save trajectory vector 
                            self.polylines_traj.append([x_start, y_start, x_end, y_end, polyline_type, timestep-self.start_time_step+1, polyline_id])

                        # pad vector if not in bounds
                        else:
                            self.polylines_traj.append([0, 0, 0, 0, 0, timestep-self.start_time_step+1, polyline_id])

                    # focal track after pred_start_timestep is future ground truth 
                    elif track.category == TrackCategory.FOCAL_TRACK:

                        # Get actor trajectory
                        x_start, y_start = rot_2D(np.array(track.object_states[timestep-1].position - self.ego_end_pos), self.ego_end_angle, self.left_handed)
                        x_end, y_end = rot_2D(np.array(track.object_states[timestep].position - self.ego_end_pos), self.ego_end_angle, self.left_handed)

                        # append ground truth
                        self.future_traj.append([x_start, y_start, x_end, y_end])

                # pad timestep if no data is available
                elif 0 < timestep < self.pred_start_timestep:

                    # get polylnie
                    polyline_id = self.get_polyline_id(track.track_id)

                    # save trajectory vector v_i as trajectory
                    self.polylines_traj.append([0, 0, 0, 0, 0, timestep-self.start_time_step+1, polyline_id])
        
    def get_lane_polylines(self):

        # go through all lanes
        for lane_segment in self.static_map.vector_lane_segments.values():

            # encode polyline type
            polyline_type_right = self.encode_type(mark_type=self.mark_type[lane_segment.right_mark_type], lane_type=self.lane_type[lane_segment.lane_type], is_intersection=int(lane_segment.is_intersection))
            polyline_type_left = self.encode_type(mark_type=self.mark_type[lane_segment.left_mark_type], lane_type=self.lane_type[lane_segment.lane_type], is_intersection=int(lane_segment.is_intersection))
            # get lane coordinates
            self.create_polyline_from_lane([lane_segment.right_lane_boundary.xyz], polyline_type_right, lane_segment.id+1000)
            self.create_polyline_from_lane([lane_segment.left_lane_boundary.xyz], polyline_type_left, lane_segment.id+1000)

        # go through all crossings
        for ped_xing in self.static_map.vector_pedestrian_crossings.values():

            # encode polyline type
            polyline_type = self.encode_type(is_crossing=1)
            # get polylines
            self.create_polyline_from_lane([ped_xing.edge1.xyz, ped_xing.edge2.xyz], polyline_type, ped_xing.id + 2000)

    def get_end_position_angle(self):

        for track in self.scenario.tracks:
            if track.category == TrackCategory.FOCAL_TRACK:
                self.ego_end_pos = np.array(track.object_states[self.pred_start_timestep-1].position)
                # self.ego_end_angle = get_angle(np.array(track.object_states[self.pred_start_timestep-2].position), self.ego_end_pos)
                self.ego_end_angle = track.object_states[self.pred_start_timestep-1].heading

    def create_polyline_from_lane(self, lanes, polyline_type, lane_id):
        
        # go through lanes
        for polyline in lanes:
            # create vecto with sliding window
            vectors = sliding_window_view(polyline[:, 0:2], (2, 2))
            # get start and end coordinats
            for vector in vectors:
                norm_vector = vector - self.ego_end_pos
                x_start, y_start = rot_2D(norm_vector.squeeze()[-2], self.ego_end_angle, self.left_handed)
                x_end, y_end = rot_2D(norm_vector.squeeze()[-1], self.ego_end_angle, self.left_handed)

                # check if cordinates are in range
                if  (-self.ego_range[0] < x_start < self.ego_range[1]) and (-self.ego_range[2] < y_start < self.ego_range[3]) and \
                    (-self.ego_range[0] < x_end < self.ego_range[1]) and (-self.ego_range[2] < y_end < self.ego_range[3]):

                    polyline_id = self.get_polyline_id(lane_id + 1000)

                    self.polylines_lane.append([x_start, y_start, x_end, y_end, polyline_type, 0, polyline_id]) # timestep 0 for all lanes

    def get_polyline_id(self, element_id):

        if element_id not in self.poly_line_ids.keys():
            self.poly_line_ids[element_id] = self.poly_line_idx
            self.poly_line_idx += 1

        return self.poly_line_ids[element_id]

    def encode_type(self, obj_type=None, mark_type=None, lane_type=None, is_intersection=None, is_crossing=None):

        encoded_type  = 0
        	
        if obj_type:        encoded_type += obj_type
        elif mark_type:     encoded_type += self.max_obj + (is_intersection*self.max_lane*self.max_mark) + ((lane_type-1)*self.max_mark) + mark_type
        elif is_crossing:   encoded_type += self.max_obj + self.max_lane*self.max_mark*2 + 1

        return encoded_type

    def create(self, scene):

        # load scenario and map with the api from the passed paths
        scenario_id = scene.stem.split("_")[-1]  # get the id of the scenario
        static_map_path = scene.parents[0] / f"log_map_archive_{scenario_id}.json"
        self.static_map = ArgoverseStaticMap.from_json(static_map_path)
        self.scenario = scenario_serialization.load_argoverse_scenario_parquet(scene)

        # init polyline vectors
        self.polylines_traj = []
        self.future_traj = []
        self.polylines_lane = []

        # init polyline id's
        self.poly_line_ids = {}
        self.poly_line_idx = 1

        # get position of target vehicle and heading at last timestep
        self.get_end_position_angle()

        # get polylines of agents and the future ground truth trajectory that has to be predicted
        self.get_agent_polylines()
        self.get_lane_polylines()

        # pad zeros if no map data is available
        if len(self.polylines_lane) == 0: self.polylines_lane.append([0, 0, 0, 0, 0, 0, 0])

        return np.array(self.polylines_traj), np.array(self.polylines_lane), np.array(self.future_traj)


#########################################################################################################################################################

def get_scenes(data_dir_raw, dataset):

    # get all file paths to data
    return sorted(Path(data_dir_raw+dataset).rglob("*.parquet"))


def get_polyline_class(input_len, pred_len, left_handed):

    # create and return polyline class
    return  Polylines(input_len, pred_len, left_handed)
