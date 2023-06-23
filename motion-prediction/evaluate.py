import os
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse

from src.utils.data_utils.dataset import PolylineDataset
from src.utils.train_eval_utils.evaluater import evaluater, get_safe_pred_horizon

########################################################################################################################

# load the config stored in config.yml
def load_config():
    with open('./config.yml', 'r') as file:
        return yaml.safe_load(file)

# load config
CONFIG = load_config()

########################################################################################################################

# Define the parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--trained_model',  type=str, default=None)
parser.add_argument('--model_path',  type=str, default=None)
parser.add_argument('--save_path',  type=str, default=None)

parser.add_argument('--small_dataset', type=float, default=CONFIG['dataset_option']['small_dataset_percent'])
parser.add_argument('--num_workers', type=int, default=CONFIG['dataset_option']['dataloader_num_workers'])
parser.add_argument('--save_prediction_horizon', type=bool, default=CONFIG['train_eval_options']['save_prediction_horizon'])
parser.add_argument('--batch_size', type=int, default=CONFIG['train_eval_options']['eval_batch_size'])
parser.add_argument('--eval_left', type=bool, default=CONFIG['train_eval_options']['eval_left_option'])
parser.add_argument('--eval_dataset', type=str, default=CONFIG['train_eval_options']['eval_dataset'])
parser.add_argument('--eval_by_direction', type=bool, default=CONFIG['train_eval_options']['eval_by_direction'])
parser.add_argument('--save_eval', type=bool, default=CONFIG['train_eval_options']['save_eval'])
parser.add_argument('--loss_function', type=str, default=CONFIG['model']['loss_function'])
parser.add_argument('--left_handed', type=bool, default=CONFIG['dataset_option']['left_handed_option'])
parser.add_argument('--left_and_right', type=bool, default=CONFIG['dataset_option']['left_and_right'])
args = parser.parse_args()

# check must have args
if not args.trained_model:
    raise Exception('Saved model must be defined with --trained_model')
if not args.model:
    raise Exception('Model must be defined with --model')
if not args.dataset:
    raise Exception('Dataset must be defined with --dataset')

model_path = args.model_path if args.model_path else './trained_models/'
save_path = args.save_path if args.save_path else './logs/evaluations/'

# set input and output lenghts
input_len = CONFIG['prediction_horizon']['input_len_sec']
pred_len = CONFIG['prediction_horizon']['pred_len_sec']

# get sample rate
if args.dataset == 'nuscenes':
    hz = 2
elif args.dataset == 'av2' or args.dataset == 'argoverse':
    hz = 10

# set directory and name
data_dir = CONFIG['directory']['path']+args.dataset+'/'+'data_'+str(input_len)+'s'+str(pred_len)+'s/'

##################################################################################################################

# Random Seed
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(CONFIG['seed'])

########################################################################################################################

# set dataset
data_set = PolylineDataset(data_dir + args.eval_dataset+'/', args.model,
                           args.small_dataset, args.left_and_right, args.left_handed)

# only use part of dataset if passed
if args.small_dataset:
    print(f'{args.small_dataset}% of dataset used')
else:
    print('100% of data used')

# dataloader
data_loader = DataLoader(data_set, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers)

# get device
if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')
elif torch.has_mps:
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Device: {device}')

########################################################################################################################

# load model
model = torch.load(model_path+args.trained_model, map_location=device)
model = model.to(device)

##################################################################################################################

# save safe_prediction_horizon if passed
if args.save_prediction_horizon:
    print(f'Getting safe prediction horizon:')
    safe_prediction_horizon = get_safe_pred_horizon(model=model,
                                                    device=device,
                                                    data_loader=data_loader,
                                                    sample_freq=hz)

    file_name = f'pred_horizon-{args.trained_model[:-4]}'
    np.save('./logs/self_eval_avs_data/'+file_name, safe_prediction_horizon)
    print(
        f'Prediction horizon data saved to : ./logs/self_eval_avs_data/{file_name}')

##################################################################################################################

# evaluate model
else:

    # list of directions to evaluate (None means all data will be evaluated)
    if args.eval_by_direction:
        directions = ["right turn", "left turn",
                      "straight drive", "other", None]
    else:
        directions = [None]

    # list of evaluation metrics
    val_loss_list, val_ade_list, val_fde_list, val_miss_rate_list, num_scenes_list = [], [], [], [], []

    # evaluate for all directions
    for direction in directions:

        print(f"Evaluating {'all' if direction==None else direction}:")
        val_loss, val_ade, val_fde, val_miss_rate, num_scenes = evaluater(model=model,
                                                                          device=device,
                                                                          data_loader=data_loader,
                                                                          direction=direction)
        # save all metrics
        val_loss_list.append(np.array(val_loss))
        val_ade_list.append(np.array(val_ade.cpu()))
        val_fde_list.append(np.array(val_fde.cpu()))
        val_miss_rate_list.append(np.array(val_miss_rate))
        num_scenes_list.append(np.array(num_scenes))

        # print metrics
        print(f"Loss:\t\t{val_loss:.3f}")
        print(f"ADE:\t\t{val_ade:.3f}")
        print(f"FDE:\t\t{val_fde:.3f}")
        print(f"Miss Rate:\t{val_miss_rate:.3f}")
        print(f"Num Scenes:\t{num_scenes}/{len(data_set)}")

    # save metrics if passed
    if args.save_eval:
        evaluation = np.concatenate(
            (val_loss_list, val_ade_list, val_fde_list, val_miss_rate_list, num_scenes_list))
        if args.left_and_right:
            file_name = f'evaluation-left_right-{args.trained_model[:-4]}'
        elif args.left_handed:
            file_name = f'evaluation-left-{args.trained_model[:-4]}'
        else:
            file_name = f'evaluation-right-{args.trained_model[:-4]}'
        np.save(save_path+file_name, evaluation)
        print(f'Evaluation metrics saved to : ./logs/evaluations/{file_name}')
