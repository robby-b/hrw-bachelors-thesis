import os
import yaml
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time
import copy

import src.utils.data_utils.preprocessing.nuscenes as nuscenes
import src.utils.data_utils.preprocessing.av2 as av2
import src.utils.data_utils.preprocessing.argoverse as argoverse

########################################################################################################################

# load the config stored in config.yml
def load_config():
    with open('./config.yml', 'r') as file:
        return yaml.safe_load(file)

# load config
CONFIG = load_config()

########################################################################################################################

# create data according to list of data
def create_data(polylines_class, scenes, data_dir, start_scenario, scenarios_per_file, left_handed):

    # copy class for each process
    polylines = copy.copy(polylines_class)

    # init index for naming the data files last data saved
    data_idx = 0
    idx_last = 0

    # go through data according to chunk size
    for idx in range(scenarios_per_file, len(scenes)+1, scenarios_per_file):

        data = []
        for scene in scenes[idx_last:idx]:

            # Get the Polyline vectors
            p_traj, p_lane, future_traj = polylines.create(scene)

            # add to data
            data.append([p_traj, p_lane, future_traj])

        # save data
        if left_handed: file_name = f"data_left_handed_{data_idx+(start_scenario//scenarios_per_file)}"
        else:           file_name = f"data_{data_idx + (start_scenario // scenarios_per_file)}"
        np.savez_compressed(data_dir+file_name, np.array(data, dtype=object))
        idx_last = idx
        data_idx += 1

# split data so it can be processed in multiple parts
def split_for_mulitprocessing(raw_data):

    # limit number of files if passed
    if dataset == 'train/':
        if args.num_scenarios_train:
            raw_data = raw_data[:args.num_scenarios_train]
    else:
        if args.num_scenarios_val:
            raw_data = raw_data[:args.num_scenarios_val]

    # throw away last few scenes to ensure uniform data length
    discarded_scenes = len(raw_data) % (args.scenarios_per_file)
    if discarded_scenes != 0:   raw_data = raw_data[:-discarded_scenes]
    print(f'Last {discarded_scenes} scenes discarded to ensure uniform data length.')

    # get number of files that have to be created and print info
    num_files = len(raw_data) // args.scenarios_per_file
    print(f'{num_files} files will be created with {args.scenarios_per_file} scenarios per file.')

    # split paths into equal sizes for every process
    scenario_step = (len(raw_data) // (num_processes * args.scenarios_per_file) + 1) * args.scenarios_per_file
    raw_data_split = [raw_data[i:i + scenario_step] for i in range(0, len(raw_data), scenario_step)]

    return raw_data_split, scenario_step, num_files

# count files and display progressbar
def count_files(directory, num_files, left_handed):

    # create progressbar
    pbar = tqdm(total=num_files, desc='processing')

    # init file counts
    count = 0
    last_count = 0

    # check file count as long as not all files have been processed
    while num_files > count:

        # reset count after last count
        count = 0
        
        # go through folder and count files
        for file_name in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file_name)): 
                if left_handed and 'data_left_handed' in file_name:     count += 1
                elif ('data' in file_name) and ('left_handed' not in file_name) and not left_handed:  count += 1

        # if count changed update progressbar
        if count > last_count:
            pbar.update(count-last_count)  
            last_count = count

        # sleep for half a second so prcess doesnt run all the time
        time.sleep(.5)

    # close progressbar when finished
    pbar.close()

# create multiprocessing processes
def create_processes(process_instance):

    print('Initialising processes that create polylines ...')
    processes = []

    # create progressbar process
    p = mp.Process(target=count_files, args=(data_dir+dataset, num_files, args.left_handed))
    p.start()
    processes.append(p)

    # create processes that create the data
    for i in range(num_processes):
        # processes cant be created if too many processes for num_scenarios -> just pass in this case
        try:
            # create a new process instance
            p =  process_instance(i)
            p.start()
            processes.append(p)
        except: pass

    # join processes so code only continues after all processes have been executed
    for process in processes:
        process.join()

# save meta data file that hold the maximum length of the data
def save_max_data_lenghts():

    # collect file names
    file_names = []
    for file_name in os.listdir(data_dir+dataset):
        if  (("data_left_handed" in file_name) and args.left_handed) or \
            (("data" in file_name) and not ("data_left_handed" in file_name) and not args.left_handed):
            file_names.append(file_name)

    # init lengths
    num_agents_max = 0
    num_polylines_max = 0
    num_vectors_max = 0

    # go through all files to get the max length of data for padding and save in meta file 
    for file_name in tqdm(file_names, desc=f"Collecting max data lengths for {data_dir+dataset} ..."):
        
        # load data (['arr_0'] as array is saved as .npz)
        data = np.load(os.path.join(data_dir+dataset, file_name), allow_pickle=True)['arr_0']

        # collect scene 
        for scene in data:
            # get input data
            features = np.concatenate((scene[0], scene[1]), axis=0)
            # get max number of polylines in current scene
            max_polylines_scene = np.max(features[:, -1])
            if  max_polylines_scene > num_polylines_max: num_polylines_max = max_polylines_scene
            # get the max number of vectors in current scene and update if necessary
            max_vectors_scene = len(features)
            if max_vectors_scene > num_vectors_max: num_vectors_max = max_vectors_scene
            # get max number of agents in current scene
            num_agents_scene = np.max(scene[0][:, -1])+1
            if  num_agents_scene > num_agents_max: num_agents_max = num_agents_scene

    # save data
    if args.left_handed:    meta_file_name = data_dir+dataset+f"meta_left_handed"
    else:                   meta_file_name = data_dir+dataset+f"meta"
    
    np.save(meta_file_name, {   'num_polylines_max':    int(num_polylines_max),
                                'num_vectors_max':      int(num_vectors_max),
                                'num_agents_max':       int(num_agents_max)})

########################################################################################################################

if __name__ == '__main__':
    
    # Define the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)

    parser.add_argument('--num_scenarios_train', type=int, default=CONFIG['preprocessing']['process_num_scenarios_train'])
    parser.add_argument('--num_scenarios_val', type=int, default=CONFIG['preprocessing']['process_num_scenarios_val'])
    parser.add_argument('--scenarios_per_file', type=int, default=CONFIG['preprocessing']['scenarios_per_file'])
    parser.add_argument('--left_handed', type=bool, default=CONFIG['dataset_option']['left_handed_option'])
    args = parser.parse_args()

    if not args.dataset: raise Exception('Dataset must be defined with --dataset')

    # set input and output lenghts
    input_len = CONFIG['prediction_horizon']['input_len_sec']
    pred_len = CONFIG['prediction_horizon']['pred_len_sec']

    # define directory
    data_dir = CONFIG['directory']['path'] + args.dataset + '/' + 'data_' + str(input_len) + 's' + str(pred_len) + 's/'
    data_dir_raw = CONFIG['directory']['path'] + args.dataset + '/' + CONFIG['directory']['raw']

    # define dataset folder names
    datasets = ['train/', 'val/']

    # get number of cpu cores (-1 as progressbar requires one process)
    num_processes = mp.cpu_count()-1

    #####################################################################################################################

    # create data for both datasets
    for dataset in datasets:

        # create folder if they dont exist
        if not os.path.exists(data_dir):                           os.mkdir(data_dir)
        if not os.path.exists(data_dir+dataset):                   os.mkdir(data_dir+dataset)

        ##################################################################################################################
        
        print(f'Collecting scenes for {dataset[:-1]} data ...')

        # Argoverse 2 dataset
        if args.dataset == 'av2':
            scenes = av2.get_scenes(data_dir_raw, dataset)
            polylines = av2.get_polyline_class(input_len, pred_len, args.left_handed)
        
        # Argoverse 1 dataset
        elif args.dataset == 'argoverse':
            scenes =  argoverse.get_scenes(data_dir_raw+dataset)
            polylines = argoverse.get_polyline_class(data_dir_raw+dataset, input_len, pred_len,  args.left_handed)

        # NuScenes dataset
        elif args.dataset == 'nuscenes':
            scenes =  nuscenes.get_scenes(data_dir_raw, dataset, args.left_handed)
            polylines = nuscenes.get_polyline_class(data_dir_raw, input_len, pred_len)

        # Dataset not implemented
        else: raise NotImplementedError

        ##################################################################################################################

        # splilt scenes so they can be processed by multiple processes
        scenes_split, scenario_step, num_files = split_for_mulitprocessing(scenes)

        # create processes
        create_processes(lambda i: mp.Process(target=create_data, args=(polylines, 
                                                                        scenes_split[i],
                                                                        data_dir+dataset,
                                                                        scenario_step*i,
                                                                        args.scenarios_per_file,
                                                                        args.left_handed)))
            
        ##################################################################################################################

        # safe meta data file
        save_max_data_lenghts()
