import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from src.utils.data_utils.dataset import PolylineDataset

from src.utils.plot_utils.plot_moving_scene import plot_moving_scene
from src.utils.plot_utils.plot_vector_map import plot_vector_map
from src.utils.plot_utils.plot_prediction import plot_prediction

########################################################################################################################

# load the config stored in config.yml


def load_config():
    with open('./config.yml', 'r') as file:
        return yaml.safe_load(file)


# load config
CONFIG = load_config()

##################################################################################################################

# Define the parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--model', type=str, default=None)

parser.add_argument('--moving_scene', type=bool,
                    default=CONFIG['plot_options']['plot_moving_scene'])
parser.add_argument('--vector_map', type=bool,
                    default=CONFIG['plot_options']['plot_vector_map'])
parser.add_argument('--vector_map_sample', type=bool,
                    default=CONFIG['plot_options']['plot_vector_map_sample'])
parser.add_argument('--prediction', type=str,
                    default=CONFIG['plot_options']['plot_prediction'])
parser.add_argument('--scene_idx', type=int,
                    default=CONFIG['plot_options']['viz_scene_idx'])
parser.add_argument('--dataset_split', type=str,
                    default=CONFIG['plot_options']['viz_dataset'])
parser.add_argument('--left_handed', type=bool,
                    default=CONFIG['dataset_option']['left_handed_option'])
args = parser.parse_args()

# check must have args
if not args.dataset:
    raise Exception('Dataset must be defined with --dataset')

# set directory
data_dir = CONFIG['directory']['path']+args.dataset+'/'+'data_' + \
    str(CONFIG['prediction_horizon']['input_len_sec'])+'s' + \
    str(CONFIG['prediction_horizon']['pred_len_sec'])+'s/'
data_dir_raw = CONFIG['directory']['path'] + \
    args.dataset+'/'+CONFIG['directory']['raw']

# set input and put lenghts
input_len = CONFIG['prediction_horizon']['input_len_sec']
pred_len = CONFIG['prediction_horizon']['pred_len_sec']

##################################################################################################################

# plot moving scene
if args.moving_scene:
    if args.dataset != 'av2':
        # this function is based on the av2 api and can therefore obly be used for the raw av2 data
        raise NotImplementedError
    plot_moving_scene(scene_idx=args.scene_idx, dataset=args.dataset_split,
                      data_dir=data_dir_raw, scene_len=input_len+pred_len)

# plot generated vector map
if args.vector_map:
    plot_vector_map(scene_idx=args.scene_idx, data_dir=data_dir +
                    args.dataset_split+'/', left_handed=args.left_handed, sample=False)

# plot random generated sample of scenes
if args.vector_map_sample:
    plot_vector_map(scene_idx=args.scene_idx, data_dir=data_dir +
                    args.dataset_split+'/', left_handed=args.left_handed, sample=True)

# plot vector map with model predictions
if args.prediction:

    # check if model name was passed
    if not args.model:
        raise Exception(
            'Model must be defined with --model when using --prediction')

    # load model and move to device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    elif torch.has_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'Device: {device}')

    model = torch.load('./trained_models/' +
                       args.prediction, map_location=device)
    model = model.to(device)

    # get dataset and dataloader
    data = PolylineDataset(data_dir+args.dataset_split +
                           '/', args.model, left_handed=args.left_handed)
    loader = DataLoader(data, batch_size=1, shuffle=False)

    # plot vector map and prediction
    plot_prediction(scene_idx=args.scene_idx, model=model, loader=loader, device=device,
                    data_dir=data_dir+args.dataset_split+'/', left_handed=args.left_handed)
