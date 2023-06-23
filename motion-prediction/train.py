import os
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from time import strftime, localtime
import argparse

from src.models import model
from src.utils.data_utils.dataset import PolylineDataset, ShuffleBatchSampler
from src.utils.train_eval_utils.trainer import trainer

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
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)

parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--weight_decay', type=float, default=None)

parser.add_argument('--small_dataset', type=float, default=CONFIG['dataset_option']['small_dataset_percent'])
parser.add_argument('--num_workers', type=int, default=CONFIG['dataset_option']['dataloader_num_workers'])
parser.add_argument('--log', type=bool, default=CONFIG['train_eval_options']['log_tensorboard'])
parser.add_argument('--finetune', type=str, default=CONFIG['train_eval_options']['finetune_model'])
parser.add_argument('--gpu', type=int, default=CONFIG['train_eval_options']['gpu_num'])
parser.add_argument('--left_handed', type=bool, default=CONFIG['dataset_option']['left_handed_option'])
parser.add_argument('--left_and_right', type=bool, default=CONFIG['dataset_option']['left_and_right'])
parser.add_argument('--loss_function', type=str, default=CONFIG['model']['loss_function'])
parser.add_argument('--num_predictions', type=int, default=CONFIG['model']['num_predictions'])
args = parser.parse_args()

# check necessary args
if not args.model: raise Exception('Model must be defined with --model')
if not args.dataset: raise Exception('Dataset must be defined with --dataset')

# set hyperparams based on model and dataset
epochs = args.epochs if args.epochs else CONFIG['hyperparameter'][args.dataset]['epochs']
batch_size = args.batch_size if args.batch_size else CONFIG['hyperparameter'][args.dataset]['batch_size']
lr = args.lr if args.lr else CONFIG['hyperparameter'][args.dataset]['lr']
weight_decay = args.weight_decay if args.weight_decay else CONFIG['hyperparameter'][args.dataset]['weight_decay']

# set input and output lenghts
input_len = CONFIG['prediction_horizon']['input_len_sec']
pred_len = CONFIG['prediction_horizon']['pred_len_sec']

# get sample rate
if args.dataset == 'nuscenes':                              hz = 2
elif args.dataset == 'av2' or args.dataset == 'argoverse':  hz = 10

# dataset size
if args.small_dataset:  dataset_percent = args.small_dataset
else:                   dataset_percent = 100.0

# set directory and name
data_dir = CONFIG['directory']['path']+args.dataset+'/'+'data_'+str(input_len)+'s'+str(pred_len)+'s/'
if args.left_and_right: ident = f'{input_len}s{pred_len}s'+'-left_right'
elif args.left_handed:  ident = f'{input_len}s{pred_len}s'+'-left'
else:                   ident = f'{input_len}s{pred_len}s'+'-right'

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

##################################################################################################################

# load datasets
train_set = PolylineDataset(data_dir+'train/', args.model, dataset_percent, args.left_and_right, args.left_handed)
val_set = PolylineDataset(data_dir+'val/', args.model, dataset_percent, args.left_and_right, args.left_handed)

# only use part of dataset if passed
print(f'{dataset_percent}% of dataset used')

# create dataloaders
sampler = ShuffleBatchSampler(np.arange(len(train_set)), batch_size)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, sampler=sampler)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

##################################################################################################################

# get device
if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')
elif torch.has_mps:
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# FIXME: There seems to be a bug with pytorch, so for now needs to be trained on cpu https://github.com/pytorch/pytorch/issues/96153
device = torch.device('cpu')
print(f'Device: {device}')

##################################################################################################################

# load model
if args.model   == 'VectorNet':     model = model.VectorNet(predicted_steps=pred_len*hz, num_predictions=args.num_predictions, loss_fun=args.loss_function)
elif args.model == 'LSTM':          model = model.LSTM(predicted_steps=pred_len*hz, num_predictions=args.num_predictions, loss_fun=args.loss_function)
elif args.model == 'VectorLSTM':    model = model.VectorLSTM(predicted_steps=pred_len*hz, num_predictions=args.num_predictions, loss_fun=args.loss_function)

##################################################################################################################

# use pretrained model and only re-train last prediction layer
if args.finetune:
    # load existing trained model
    model = torch.load('./trained_models/'+args.finetune)
    # add string to ident that indicated what model was finetuned
    ident = 'finetuned_'+str.split(args.finetune, '-')[-2]+'_'+str.split(args.finetune, '-')[-1][:-4]+'-'+ident
    # set learning rate to 10% of default lr for finetuning
    lr *= 0.1

##################################################################################################################

# move model to available device
model = model.to(device)

##################################################################################################################

# get current time and update tensorboard log directory
time_str = f'{strftime("%Y-%m-%d_%H-%M", localtime())}'
if args.log:
    log_dir = f'./logs/tensorboard/{args.model}-{args.loss_function}-K{args.num_predictions}-{args.dataset}-{ident}-{dataset_percent}-{time_str}'+'/'
    print(f'Tensorboard logs will be saved to: {log_dir}')
else:
    log_dir = None

##################################################################################################################

# train model
print(f'Start training on dataset {data_dir} ...')
start_timestamp = time.time()
trainer(model=model,
        device=device,
        data_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        epochs=epochs,
        weight_decay=weight_decay,
        log_dir=log_dir)
time_passed = time.time()-start_timestamp
print(f'... training finished in {time_passed:.0f}s.')

##################################################################################################################

# save model
model_name = f'{args.model}-{args.loss_function}-K{args.num_predictions}-{args.dataset}-{ident}-{dataset_percent}.pth'
torch.save(model, './trained_models/'+model_name)
print(f'Model saved to: ./trained_models/{model_name}')
