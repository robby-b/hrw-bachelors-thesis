import torch
import numpy as np
import os
import os.path as osp
from collections import Counter

from torch.utils.data import Dataset

########################################################################################################################

# Dataset for preprocessed polyline data
class PolylineDataset(Dataset):
    def __init__(self, directory, model, length_percent=None, left_and_right=False, left_handed=False):

        # get model name 
        self.model = model

        # length of data used in percent
        self.length_percent = length_percent

        # left handed option
        self.left_handed = left_handed
        self.left_and_right = left_and_right

        # get data filenames
        self.processed_dir = directory
        self.processed_paths = []
        self.get_file_names()

        # if no data available -> create data first
        if len(self.processed_paths) == 0:
            raise Exception("No data available!")

        # get meta data an max lengths
        if left_and_right:
            meta_data1 = np.load(self.processed_dir+'meta.npy', allow_pickle=True).item()
            meta_data2 = np.load(self.processed_dir+'meta_left_handed.npy', allow_pickle=True).item()
            self.num_polylines_max = max(meta_data1['num_polylines_max'], meta_data2['num_polylines_max'])
            self.num_vectors_max = max(meta_data1['num_vectors_max'], meta_data2['num_vectors_max'])
            self.num_agents_max = max(meta_data1['num_agents_max'], meta_data2['num_agents_max'])
        else:
            if left_handed: file_name = 'meta_left_handed.npy'
            else:           file_name = 'meta.npy'    
            meta_data = np.load(self.processed_dir+file_name, allow_pickle=True).item()
            self.num_polylines_max = meta_data['num_polylines_max']
            self.num_vectors_max = meta_data['num_vectors_max']
            self.num_agents_max = meta_data['num_agents_max']

        # remember last file and data to avoid reloading
        self.last_file_idx = 0
        self.data = np.load(self.processed_paths[0], allow_pickle=True)['arr_0']

        # get len of dataset and file
        self.file_size = len(self.data)
        self.length = len(self.processed_paths)*self.file_size

    def get_file_names(self):

        # get all files in data directory
        for file_name in os.listdir(self.processed_dir):
            if self.left_and_right and ('data' in file_name):
                self.processed_paths.append(osp.join(self.processed_dir, file_name))
            else:
                if self.left_handed and 'data_left_handed' in file_name: 
                    self.processed_paths.append(osp.join(self.processed_dir, file_name))
                elif ('data' in file_name) and ('left_handed' not in file_name) and not self.left_handed: 
                    self.processed_paths.append(osp.join(self.processed_dir, file_name))

        # shuffle paths but safe sorted array
        self.sorted_paths = self.processed_paths.copy()
        np.random.shuffle(self.processed_paths)

        # only use small part of data if passed
        if self.length_percent:
            last_value = int(np.ceil(len(self.processed_paths) / 100 * self.length_percent))
            self.processed_paths = self.processed_paths[:last_value]

    def get_raw(self, idx):
        # this function is used mainly if data has to be plotted

        # get file and index of indexed scene and return raw data
        data_idx = idx%self.file_size
        file_idx = idx//self.file_size
        raw_data = np.load(self.sorted_paths[file_idx], allow_pickle=True)['arr_0'][data_idx]

        return raw_data[0], raw_data[1], raw_data[2]

    def get_x_graph(self, features):

        # get polyline data without id and cluster (list of polyline ids)
        x = np.array(features[:, :-1])
        cluster = np.array(features[:, -1])

        # get num polylines in current scene
        num_polylines_scene = int(np.max(cluster))

        # add missing polylines to cluster by paddig missing range
        cluster = np.hstack([cluster, np.arange(num_polylines_scene, self.num_polylines_max)])

        # pad input and cluster with zeros so all data has uniform length
        x = np.pad(x, ((0, self.num_vectors_max+self.num_polylines_max-len(x)), (0, 0)))
        cluster = np.pad(cluster, (0, len(x)-len(cluster)))
            
        # return np array of input data
        return torch.tensor(x).float(), torch.tensor(cluster).long()

    def get_x_agents(self, features):

        # pad and expand data to shape [num_agents, seq_len, features]
        x = features
        x = np.take_along_axis(x, np.expand_dims(np.argsort(x[:,-1]), -1).repeat(x.shape[-1], 1), axis=0)               # sort according to polylines
        x = np.split(x, np.cumsum(list(Counter(x[:,-1].tolist()).values()))[:-1])                                       # split after each polyline (discard last empty split)
        x = [np.take_along_axis(p[:,:4], np.expand_dims(np.argsort(p[:,-2]), -1).repeat(4, 1), axis=0) for p in x]      # sort along timestep within each polyline (discard timestep and polyline feature)
        x = np.pad(x, ((0, self.num_agents_max-len(x)), (0,0), (0,0)))                                                  # pad missing agents
        
        # return float tensor
        return  torch.tensor(x).float()
    
    def get_y_diff(self, gt):

        # calculate the change for every timestep and return as float tensor
        gt_offset = np.vstack([gt[:, 2] - gt[:, 0],  gt[:, 3] - gt[:, 1]]).reshape(-1)
        return torch.tensor(gt_offset).float()

    def __getitem__(self, idx):
        
        # get file and index of indexed scene
        data_idx = idx%self.file_size
        file_idx = idx//self.file_size

        # get scene from correct file if new file is indexed
        if self.last_file_idx != file_idx:
            self.last_file_idx = file_idx
            self.data = np.load(self.processed_paths[file_idx], allow_pickle=True)['arr_0']

        # load the scene and get data
        scene = self.data[data_idx]
        agent_data = scene[0]
        map_data = scene[1]
        agent_map_data = np.concatenate((agent_data, map_data), axis=0)
        gt = scene[2]

        # get data according to model
        if self.model == 'VectorNet':   
            x, cluster = self.get_x_graph(agent_map_data)
            y = self.get_y_diff(gt)
            data = [x, cluster, y]

        elif self.model == 'LSTM':       
            x = self.get_x_agents(agent_data)
            y = self.get_y_diff(gt)
            data = [x, y]

        elif self.model == 'VectorLSTM': 
            x_agents = self.get_x_agents(agent_data)
            x_map, cluster_map = self.get_x_graph(map_data)
            x_map = x_map[:,:-1] # timesteps are not relevant for map if processed seperatly
            y = self.get_y_diff(gt)
            data = [x_agents, x_map, cluster_map, y]

        else: raise NotImplementedError

        # return list of data
        return data

    def __len__(self):
        return self.length

########################################################################################################################

# Sampler that only shuffles batches -> used to avoid constant loading of files when accessing dataset randomly
class ShuffleBatchSampler():
    def __init__(self, data_indices, batch_size):

        self.data_indices = data_indices
        self.batch_size = batch_size
        self.len = len(data_indices)

    def __iter__(self):
        # split data into arrays of batch_size
        batch_array = np.array_split(self.data_indices, len(self.data_indices) // self.batch_size)
        # update random seed so new epoch is shuffled different and shuffle batches
        np.random.seed(np.random.get_state()[1][0] + 1)
        np.random.shuffle(batch_array)
        # concat data again to 1D idx array
        output_indices = np.concatenate(batch_array)
        return iter(output_indices)

    def __len__(self):
        return self.len
