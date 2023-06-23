import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import os.path as osp

from src.utils.data_utils.dataset import PolylineDataset


def plot_vector_map(scene_idx, data_dir, left_handed=False, show=True, sample=False,):

    # fov limit
    limit_fov = 60

    # get Polylines
    Polylines = PolylineDataset(directory=data_dir, model=None, left_handed=left_handed)

    # plot get random samples
    if sample:  
        # get all files in data directory
        processed_paths = []
        for file_name in os.listdir(data_dir):
            if left_handed and 'data_left_handed' in file_name: 
                processed_paths.append(osp.join(data_dir, file_name))
            elif ('data' in file_name) and ('left_handed' not in file_name) and not left_handed: 
                processed_paths.append(osp.join(data_dir, file_name))
        file_size = len(np.load(processed_paths[0], allow_pickle=True)['arr_0'])

        # get number of scenes
        scenes = np.random.randint(file_size*len(processed_paths), size=9)
    else:
        scenes = [scene_idx]

    # plot all scenes
    figure = plt.figure()
    for i, scene in enumerate(scenes):

        # get data
        p_traj, p_lane, future_traj = Polylines.get_raw(scene)

        # make subplot if samples should be plotted
        if sample:
            plt.subplot(3, 3, i+1)
            plt.title(i+1)

        # create plot
        for v in p_traj:
            if v[-1] == 0:  color = 'orange'
            else:           color = 'blue'
            plt.quiver(v[0], v[1], v[2]-v[0], v[3]-v[1],
                        angles='xy', scale_units='xy', scale=1, headwidth=3, width=0.001, color=color)
        for v in p_lane:
            plt.quiver(v[0], v[1], v[2]-v[0], v[3]-v[1],
                        angles='xy', scale_units='xy', scale=1, headwidth=3, width=0.001, color='black')
        for v in future_traj:
            plt.quiver(v[0], v[1], v[2]-v[0], v[3]-v[1],
                        angles='xy', scale_units='xy', scale=1, headwidth=3, width=0.001, color='green')
            
        # limit field of view 
        plt.xlim([-limit_fov, limit_fov])
        plt.ylim([-limit_fov, limit_fov])

    # add legend and show plot
    if show: 
        # Creating legend with color box
        gt = mpatches.Patch(color='green', label='ground truth future')
        focal_track = mpatches.Patch(color='orange', label='ego agent past')
        agents = mpatches.Patch(color='blue', label='agents past')
        lanes = mpatches.Patch(color='black', label='lanes')
        figure.legend(handles=[gt, focal_track, agents, lanes])
        plt.show()  
