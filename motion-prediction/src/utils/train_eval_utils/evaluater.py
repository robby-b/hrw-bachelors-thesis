import torch
from tqdm import tqdm
import src.utils.train_eval_utils.metrics as metrics
import src.utils.train_eval_utils.helper as helper

########################################################################################################################

# loop through data and evaluate metrics
def evaluater(model, device, data_loader, direction=None):

    # inint running metrics
    running_loss= 0.0
    running_ade = 0.0
    running_fde = 0.0
    running_miss_rate = 0.0
    evaluated_batches = 0
    evaluated_scenes = 0

    # loop through batches
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="evaluating model"):

            # get data and move them to device
            data = [data.to(device) for data in batch]
            input_data, y = data[:-1], data[-1]

            # pass data through model
            pred, loss = model.predict_best(input_data, y, return_loss=True)

            # calc metrics for current batch
            batch_loss = loss.item()
            batch_ade, batch_fde, batch_miss_rate, evaluated_scenes_batch = eval_batch(pred, y, direction)

            # add to running total (only loss and ade when training to safe time)
            if evaluated_scenes_batch != 0:
                evaluated_batches += 1 
                evaluated_scenes += evaluated_scenes_batch 
                running_loss += batch_loss
                running_ade += batch_ade
                running_fde += batch_fde
                running_miss_rate += batch_miss_rate
    
    # calc averages
    loss = running_loss/evaluated_batches
    ade = running_ade/evaluated_batches
    fde = running_fde/evaluated_batches
    miss_rate = running_miss_rate/evaluated_batches

    # return mean values for metrics
    return loss, ade, fde, miss_rate, evaluated_scenes

########################################################################################################################

# evaluate metrics for a given batch
def eval_batch(out, gt, direction=None):

    # init running metrics
    running_ade_batch = 0.0
    running_fde_batch = 0.0
    total_missed = 0
    skipped = 0

    # calc metrics for every scene
    for scene_idx in range(gt.size(0)):

        # get coordinates
        x = helper.format_to_coordinates(out[scene_idx])
        y = helper.format_to_coordinates(gt[scene_idx])
        
        # check direction and calc metrics
        if direction == None or direction == helper.get_direction(y):
            displacements = metrics.calc_l2(x, y)
            running_ade_batch += metrics.get_ade(displacements)
            running_fde_batch += metrics.get_fde(displacements)
            total_missed +=  metrics.get_miss(displacements, threshold=2.0)
        else:
            skipped += 1

    # get number of scenes that were evaluated
    valid_scenes = scene_idx+1-skipped

    # calc average (check if no scene for condition was found in batch)
    ade_batch = running_ade_batch/valid_scenes if valid_scenes != 0 else 0
    fde_batch = running_fde_batch/valid_scenes if valid_scenes != 0 else 0
    missed_batch = total_missed/valid_scenes if valid_scenes != 0 else 0

    # return mean values for metrics
    return ade_batch, fde_batch, missed_batch, valid_scenes

########################################################################################################################

# get save prediction horizon (last timestep where displacement < 2m) over last observed velocity
def get_safe_pred_horizon(model, device, data_loader, sample_freq):

    # inint 
    safe_prediction_horizon = torch.empty(0)

    # loop through batches
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="evaluating model"):

            # get data and move them to device
            data = [data.to(device) for data in batch]
            input_data, gt = data[:-1], data[-1]

            # pass data through model
            pred, _ = model.predict_best(input_data, gt, return_loss=True)

            # calc metrics for current batch
            idx_safe_batch = []
            for scene_idx in range(gt.size(0)):

                # get coordinates
                x = helper.format_to_coordinates(pred[scene_idx])
                y = helper.format_to_coordinates(gt[scene_idx])

                # calc all metrics based on l2 distance (only loss and ade when training so safe time)
                displacements = metrics.calc_l2(x, y)
                idx_last_safe = metrics.get_last_safe_idx(displacements, threshold=2.0)
                idx_safe_batch.append(idx_last_safe)

            # create tensor for save velocitys over timesteps and append 
            last_vel_batch = helper.get_last_vel(input_data, sample_freq)
            last_safe_timesteps_batch =  torch.tensor(idx_safe_batch)//sample_freq
            safe_pred_horizon_batch = torch.column_stack([last_vel_batch, last_safe_timesteps_batch])
            safe_prediction_horizon = torch.cat([safe_prediction_horizon, safe_pred_horizon_batch])
            
    return safe_prediction_horizon
