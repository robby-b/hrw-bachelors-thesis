import torch

# calc the l2 distance
def calc_l2(x, y):
    return torch.linalg.norm(x.unsqueeze(dim=0) - y, axis=2)

# get average displacemnt error
def get_ade(displacements):
    return displacements.mean()

# get the fde
def get_fde(displacements):
    return displacements[:, -1].squeeze()

# get miss
def get_miss(displacements, threshold):
    if displacements[:, -1] > threshold:    return 1
    else:                                   return 0

# get index of last safe timestep
def get_last_safe_idx(displacements, threshold):
    safe_idx_list = (displacements.squeeze() < threshold).nonzero()   # get list of all timesteps where displacement is below threshold
    if safe_idx_list.shape[0] == 0:
        return 0                                        # no safe displacement for predicted trajectory
    else:                      
        # ToDo: make loop without break (was just preliminary solution)
        for i, idx, in enumerate(torch.arange(torch.max(safe_idx_list.squeeze())+1)):
            if safe_idx_list[i] != idx: break
        return i
        