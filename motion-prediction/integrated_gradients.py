# %%
# Imports

import os
import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
import cv2

from src.utils.data_utils.dataset import PolylineDataset
from src.utils.train_eval_utils.evaluater import evaluater, get_safe_pred_horizon

# %%
# Configuration

def load_config():
    with open('./config.yml', 'r') as file:
        return yaml.safe_load(file)

# load config
CONFIG = load_config()

input_len = CONFIG['prediction_horizon']['input_len_sec']
pred_len = CONFIG['prediction_horizon']['pred_len_sec']

parser = argparse.ArgumentParser()
parser.add_argument('--trained_model', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--idx', type=int, default=None)
parser.add_argument('--type', type=str, default=None)
args_cli = parser.parse_args()


args = dict()
args['trained_model'] = args_cli.trained_model if args_cli.trained_model else "model-2s6s-LSTM-MSE-K1-nuscenes-100-2023-04-02_16-46.pth"
args['dataset'] = args_cli.dataset if args_cli.dataset else "nuscenes"
args['model'] = args_cli.model if args_cli.model else "LSTM"

args['small_dataset'] = CONFIG['dataset_option']['small_dataset_percent']
args['num_workers'] = CONFIG['dataset_option']['dataloader_num_workers']
args['save_prediction_horizon'] = CONFIG['train_eval_options']['save_prediction_horizon']

args['eval_left'] = CONFIG['train_eval_options']['eval_left_option']
args['eval_dataset'] = CONFIG['train_eval_options']['eval_dataset']
args['eval_by_direction'] = CONFIG['train_eval_options']['eval_by_direction']
args['save_eval'] = CONFIG['train_eval_options']['save_eval']
args['loss_function'] = CONFIG['model']['loss_function']
args['left_handed'] = CONFIG['dataset_option']['left_handed_option']
args['left_and_right'] = CONFIG['dataset_option']['left_and_right']


data_dir = CONFIG['directory']['path']+args['dataset'] + \
    '/'+'data_'+str(input_len)+'s'+str(pred_len)+'s/'

model_path = args['model_path'] if 'model_path' in args else './trained_models/'
save_path = args['save_path'] if 'save_path' in args else './logs/evaluations/'

batch_size = args_cli.batch_size if args_cli.batch_size else CONFIG['train_eval_options']['eval_batch_size']
arg_idx = args_cli.idx if args_cli.idx else None

# Random Seed
# np.random.seed(CONFIG['seed'])
# torch.manual_seed(CONFIG['seed'])
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(CONFIG['seed'])
#     torch.cuda.manual_seed_all(CONFIG['seed'])
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
# os.environ["PYTHONHASHSEED"] = str(CONFIG['seed'])

# %%
# set dataset
data_set = PolylineDataset(data_dir + args['eval_dataset']+'/', args['model'],
                           args['small_dataset'], args['left_and_right'], args['left_handed'])

# only use part of dataset if passed
if args['small_dataset']:
    print(f'{args["small_dataset"]}% of dataset used')
else:
    print('100% of data used')

# dataloader
data_loader = DataLoader(data_set, batch_size=batch_size,
                         shuffle=False, num_workers=args['num_workers'])

# get device
if torch.cuda.is_available():
    device = torch.device(f'cuda:{args["gpu"]}')
else:
    device = torch.device('cpu')

print(f'Device: {device}')

# %%
# load model
model = torch.load(model_path+args['trained_model'], map_location=device)
model = model.to(device)


# %%
# Evaluation

# from src.utils.train_eval_utils.evaluater import evaluater

# print(f"Evaluating:")
# val_loss, val_ade, val_fde, val_miss_rate, num_scenes = evaluater(model=model,
#                                                                 device=device,
#                                                                 data_loader=data_loader)

# # print metrics
# print(f"Loss:\t\t{val_loss:.3f}")
# print(f"ADE:\t\t{val_ade:.3f}")
# print(f"FDE:\t\t{val_fde:.3f}")
# print(f"Miss Rate:\t{val_miss_rate:.3f}")
# print(f"Num Scenes:\t{num_scenes}/{len(data_set)}")

# %%
import src.utils.train_eval_utils.metrics as metrics
import src.utils.train_eval_utils.helper as helper
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_relevant_batch_id(attributions):
    # get batch id with highest attribution
    for i in range(attributions.shape[0]):
        if torch.sum(attributions[i, None, :, :]) != 0:
            batch_id = i
            break

    return batch_id


def plot_attributions(attribution, input_tensor, type, idx=None):
    assert type == "single" or type == "batch" or type == "all", "type must be single, batch or all"

    title_text = f"IGs for {type}"
    if type == "single":
        title_text += f" {idx}"
    elif type == "batch":
        title_text += f" {idx} batch_size-{input_tensor.shape[0]}"

    time_steps = np.linspace(-2, 0, 3) # TODO: parametrize
    feature_names = ["x_start", "y_start", "x_end", "y_end"]

    plot_data = np.flip(attribution.detach().cpu().numpy())

    fig = plt.figure(figsize=(14,7))
    ax = plt.axes()
    im = plt.imshow(plot_data,
                    interpolation='nearest',
                    aspect='auto', cmap="PiYG", norm=mpl.colors.CenteredNorm())

    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])

    plt.colorbar(im, cax=cax)
    ax.set_xticks(np.arange(len(feature_names)), labels=feature_names)
    ax.set_yticks(np.arange(len(time_steps)), labels=time_steps)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Time (past)")

    # Loop over data dimensions and create text annotations.
    for i in range(len(time_steps)):
        for j in range(len(feature_names)):
            if type == "single":
                label_text = f"Coord: {np.flip(input_tensor[0, 0, :, :].detach().cpu().numpy())[i, j]:.7f}\nAttrib: {plot_data[i, j]:.7f}"
            else:
                label_text = f"Attrib: {plot_data[i, j]:.7f}"

            text = ax.text(j, i, label_text,
                        ha="center", va="center", color="black")

    ax.set_title(title_text)
    plt.savefig(f"../results/integrated_gradients/final/{title_text}.png")
    plt.show()


def ig_all(model, device, data_loader):
    batch = None

    all_attributions = []
    counter = 0
    for batch in tqdm(data_loader):
        data = [data.to(device) for data in batch]
        input_data, target = data[:-1], data[-1]
        pred = model.predict_best(input_data, target)

        input_tensor = input_data[0]
        input_tensor.requires_grad = True

        ig = IntegratedGradients(model.predict_single_coordinate, multiply_by_inputs=True)

        batch_attribution_list = []

        relevant_batch_id = None

        for batch_element in tqdm(target):
            for i in range(len(batch_element)):
                attributions = ig.attribute(inputs=input_tensor, target=None, additional_forward_args=(target, i))

                if relevant_batch_id is None:
                    relevant_batch_id = get_relevant_batch_id(attributions)

                batch_attribution_list.append(attributions[relevant_batch_id, 0, :, :]) # only the first agent count and batch size is 1

        mean = torch.mean(torch.stack(batch_attribution_list), 0)
        all_attributions.append(mean)
        counter += 1

    all_mean = torch.mean(torch.stack(all_attributions), 0)
    plot_attributions(all_mean, input_tensor, "all")

def ig_batch(model, device, data_loader, idx):
    counter = 0
    batch = None

    for b in data_loader:
        if counter == idx:
            batch = b
            break

        counter += 1

    assert batch is not None, "idx out of range"

    data = [data.to(device) for data in batch]
    input_data, target = data[:-1], data[-1]
    pred = model.predict_best(input_data, target)

    input_tensor = input_data[0]
    input_tensor.requires_grad = True

    ig = IntegratedGradients(model.predict_single_coordinate, multiply_by_inputs=True)

    attribution_list = []

    for batch_element in tqdm(target):
        for i in range(len(batch_element)):
            attributions = ig.attribute(inputs=input_tensor, target=None, additional_forward_args=(target, i))
            relevant_batch_id = get_relevant_batch_id(attributions)
            attribution_list.append(attributions[relevant_batch_id, 0, :, :]) # only the first agent count and batch size is 1

    mean = torch.mean(torch.stack(attribution_list), 0)
    plot_attributions(mean, input_tensor, "batch", idx)


def ig_single(model, device, data_loader, idx):
    data = data_loader.dataset[idx]
    input_data, target = data[:-1], data[-1]

    input_data[0] = input_data[0][None, :]
    target = target[None, :]

    pred = model.predict_best(input_data, target)

    input_tensor = input_data[0]
    input_tensor.requires_grad = True

    ig = IntegratedGradients(model.predict_single_coordinate, multiply_by_inputs=True)
    print(target.shape)

    attribution_list = []
    for i in range(len(target)):
        attributions = ig.attribute(inputs=input_tensor, target=None, additional_forward_args=(target, i))
        attribution_list.append(attributions[0, 0, :, :]) # only the first agent count and batch size is 1

    mean = torch.mean(torch.stack(attribution_list), 0)

    plot_attributions(mean, input_tensor, "single", idx)


def ig_evaluate(model, device, data_loader, eval_type, idx=None):
    # loop through batches
    model.eval()

    if eval_type == "rand_single":
        # Plot with coordinates
        assert idx is None, "idx should be None"
        idx = np.random.randint(0, len(data_loader.dataset))
        ig_single(model, device, data_loader, idx)
    elif eval_type == "rand_batch":
        assert idx is None, "idx should be None"
        idx = np.random.randint(0, len(data_loader))
        ig_batch(model, device, data_loader, idx)
        # Random batch
    elif eval_type == "single":
        assert idx is not None, "idx should not be None"
        ig_single(model, device, data_loader, idx)
        # single scene
    elif eval_type == "batch":
        assert idx is not None, "idx should not be None"
        ig_batch(model, device, data_loader, idx)
        # specific batch
    elif eval_type == "all":
        assert idx is None, "idx should be None"
        ig_all(model, device, data_loader)

if args_cli.type == "single":
    if arg_idx == None:
        ig_evaluate(model, device, data_loader, "rand_single")
    else:
        ig_evaluate(model, device, data_loader, "single", arg_idx)
elif args_cli.type == "batch":
    if arg_idx == None:
        ig_evaluate(model, device, data_loader, "rand_batch")
    else:
        ig_evaluate(model, device, data_loader, "batch", arg_idx)
elif args_cli.type == "all":
    ig_evaluate(model, device, data_loader, "all")

