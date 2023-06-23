import numpy as np
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory


# Function for ploting boxes for actors
def plot_actor_bounding_box(ax, cur_location, heading, color, bbox_size):

    # load dimensions
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    # plot box that represents agents
    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y), bbox_length, bbox_width, np.degrees(heading), color=color, zorder=100)
    ax.add_patch(vehicle_bounding_box)


# Function for ploting the static map
def plot_static_map_elements(static_map: ArgoverseStaticMap) -> None:

    # Plot drivable areas
    for drivable_area in static_map.vector_drivable_areas.values():
        for polygon in [drivable_area.xyz]:
            plt.fill(polygon[:, 0], polygon[:, 1], color="#7A7A7A", alpha=0.5)

    # Plot lane segments
    for lane_segment in static_map.vector_lane_segments.values():
        for polyline in [lane_segment.left_lane_boundary.xyz, lane_segment.right_lane_boundary.xyz]:
            plt.plot(polyline[:, 0], polyline[:, 1], "-",
                     linewidth=0.5, color='black', alpha=1.0)

    for ped_xing in static_map.vector_pedestrian_crossings.values():
        for polyline in [ped_xing.edge1.xyz, ped_xing.edge2.xyz]:
            plt.plot(polyline[:, 0], polyline[:, 1], "-",
                     linewidth=1.0, color='gray', alpha=1.0)


# Function for plotting the agents and trajectories
def plot_actor_tracks(ax, scenario, timestep):

    for track in scenario.tracks:
        # Get timesteps for which actor data is valid
        actor_timesteps = np.array(
            [object_state.timestep for object_state in track.object_states if object_state.timestep <= timestep])
        if actor_timesteps.shape[0] >= 1 and actor_timesteps[-1] == timestep:

            # Get actor trajectory and heading history
            actor_trajectory = np.array(
                [list(object_state.position) for object_state in track.object_states if object_state.timestep <= timestep])
            actor_headings = np.array(
                [object_state.heading for object_state in track.object_states if object_state.timestep <= timestep])

            # select track color
            if      track.category == TrackCategory.FOCAL_TRACK:    track_color = 'orange'
            else:                                                   track_color = 'blue'


            # Plot polyline for focal agent location history
            for polyline in [actor_trajectory]:
                plt.plot(polyline[:, 0], polyline[:, 1], "-", linewidth=2, color=track_color, alpha=1.0)
                

            # Plot bounding boxes for all vehicles and cyclists
            if track.object_type == ObjectType.VEHICLE:
                plot_actor_bounding_box(ax, actor_trajectory[-1], actor_headings[-1], track_color, (4.0, 2.0))

            elif track.object_type == ObjectType.CYCLIST or track.object_type == ObjectType.MOTORCYCLIST:
                plot_actor_bounding_box(ax, actor_trajectory[-1], actor_headings[-1], track_color, (2.0, 0.7),)

            else:
                plt.plot(actor_trajectory[-1, 0], actor_trajectory[-1,1], "o", color=track_color, markersize=4)


# Function to Plot the Moving Scene
def plot_moving_scene(scene_idx, data_dir, scene_len, dataset='test'):

    # Load all scenario files in the directory
    print(f'Searching for scene {scene_idx} in data ...')
    all_scenario_files = sorted(Path(data_dir+dataset).rglob("*.parquet"))
    scenario_path = all_scenario_files[scene_idx]  # select a scene
    scenario_id = scenario_path.stem.split("_")[-1]  # get the id of the scenario
    # get the path to the corresponding map
    static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
    # get map and scenario
    static_map = ArgoverseStaticMap.from_json(static_map_path)
    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)

    # create plot
    fig = plt.figure()
    ax = fig.add_subplot()

    fig.canvas.draw()
    plt.show(block=False)

    # go throuh scene time steps
    for timestep in range(scene_len*10): # av2 is sampled with 10Hz
        # clear plot window
        plt.cla()
        # Plot static map elements and actor tracks
        plot_static_map_elements(static_map)
        plot_actor_tracks(ax, scenario, timestep)
        # update plot
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    
    plt.show()
