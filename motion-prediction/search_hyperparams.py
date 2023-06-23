import os
import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import optuna
import argparse
from tqdm import tqdm
from time import strftime, localtime

from src.models import model
from src.utils.data_utils.dataset import PolylineDataset, ShuffleBatchSampler
from src.utils.train_eval_utils.helper import format_to_coordinates
import src.utils.train_eval_utils.metrics as metrics

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
parser.add_argument('--gpu', type=int, default=CONFIG['train_eval_options']['gpu_num'])
parser.add_argument('--left_handed', type=bool, default=CONFIG['dataset_option']['left_handed_option'])
parser.add_argument('--left_and_right', type=bool, default=CONFIG['dataset_option']['left_and_right'])
parser.add_argument('--loss_function', type=str, default=CONFIG['model']['loss_function'])
parser.add_argument('--num_predictions', type=int, default=CONFIG['model']['num_predictions'])

parser.add_argument('--num_trials', type=int, default=CONFIG['optuna']['num_trials'])
args = parser.parse_args()

# check must have params
if not args.model: raise Exception('Model must be defined with --model')
if not args.dataset: raise Exception('Dataset must be defined with --dataset')

# set hyperparams based on model and dataset
epochs = args.epochs if args.epochs else CONFIG['hyperparameter'][args.dataset]['epochs']
batch_size = args.batch_size if args.batch_size else CONFIG['hyperparameter'][args.dataset]['batch_size']
lr = args.lr if args.lr else CONFIG['hyperparameter'][args.dataset]['lr']
weight_decay = args.weight_decay if args.weight_decay else CONFIG['hyperparameter'][args.dataset]['weight_decay']

# set input and put lenghts
input_len = CONFIG['prediction_horizon']['input_len_sec']
pred_len = CONFIG['prediction_horizon']['pred_len_sec']

# get sample rate
if args.dataset == 'nuscenes':                              hz = 2
elif args.dataset == 'av2' or args.dataset == 'argoverse':  hz = 10

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

##################################################################################################################

# load datasets
train_set = PolylineDataset(data_dir+'train/', args.model, args.small_dataset, args.left_and_right, args.left_handed)
val_set = PolylineDataset(data_dir+'val/', args.model, args.small_dataset, args.left_and_right, args.left_handed)

# only use part of dataset if passed
if args.small_dataset:  print(f'{args.small_dataset}% of dataset used')
else:                   print('100% of dataset used')

##################################################################################################################

# get device
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

##################################################################################################################

# load model
if args.model   == 'VectorNet': model = model.VectorNet(predicted_steps=pred_len*hz, num_predictions=args.num_predictions, loss_fun=args.loss_function)
elif args.model == 'LSTM':      model = model.LSTM(predicted_steps=pred_len*hz, num_predictions=args.num_predictions, loss_fun=args.loss_function)

# move model to available device
model = model.to(device)

##################################################################################################################

# Trial function
print('Start Trial ...')
def objective(trial):

    # define trial for hyperparams
    epochs = trial.suggest_int('epochs', 50, 250)
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)

    # create dataloaders
    sampler = ShuffleBatchSampler(np.arange(len(train_set)), batch_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # set optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler =  optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    ##################################################################################################################

    # start training
    for epoch in tqdm(range(epochs), desc="epochs"):
        model.train()
        for batch in train_loader:

            # get data and move them to device
            data = [data.to(device) for data in batch]
            input_data, y = data[:-1], data[-1]

            # make optimization step
            optimizer.zero_grad()
            _, loss = model.predict_best(input_data, y, return_loss=True)
            loss.backward()
            optimizer.step()

            # update learning rate
            scheduler.step()

        ##################################################################################################################

        # get metrics for val set
        model.eval()
        with torch.no_grad():
            running_ade = 0.0
            for i, batch in enumerate(val_loader):

                # get data and move them to device
                data = [data.to(device) for data in batch]
                input_data, y = data[:-1], data[-1]

                # pass data through model and get best prediction
                out = model.predict_best(input_data, y)

                # calc metrics for every scenes
                running_ade_batch = 0.0
                for scene_idx in range(y.size(0)):

                    # get coordinates
                    x = format_to_coordinates(out[scene_idx])
                    gt = format_to_coordinates(y[scene_idx])

                    # calc all metrics based on l2 distance (only loss and ade when training so safe time)
                    displacements = metrics.calc_l2(x, gt)
                    running_ade_batch += metrics.get_ade(displacements)

                running_ade += running_ade_batch / (scene_idx + 1)

        val_ade = running_ade / (i + 1)

        ##################################################################################################################

        # check if prune trial
        trial.report(val_ade, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_ade

##################################################################################################################

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=args.num_trials)

##################################################################################################################

# get results
pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

# write results to file
time_str = f'{strftime("%Y-%m-%d_%H-%M", localtime())}'
file_name = f'./logs/hyperparam_search_{args.dataset}_{args.model}_{time_str}.txt'
with open(file_name, 'w') as outfile:
    
    outfile.write(f"Study statistics: \n")
    outfile.write(f"  Number of finished trials: {len(study.trials)}\n")
    outfile.write(f"  Number of pruned trials: {len(pruned_trials)}\n")
    outfile.write(f"  Number of complete trials: {len(complete_trials)}\n")
    outfile.write(f"Best trial: \n")
    trial = study.best_trial
    outfile.write(f"  Value: {trial.value}\n")
    outfile.write(f"  Params: \n")
    for key, value in trial.params.items():
        outfile.write(f"    {key}: {value}\n")
        