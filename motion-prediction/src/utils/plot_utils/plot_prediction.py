import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

from src.utils.plot_utils.plot_vector_map import plot_vector_map
import src.utils.train_eval_utils.metrics as metrics
import src.utils.train_eval_utils.helper as helper


def plot_prediction(scene_idx, model, loader, device, data_dir, left_handed):

    # go throug data and find scene
    with torch.no_grad():
        scene_found = False
        for i, batch in enumerate(loader):
            if i == scene_idx:
                scene_found = True

                # load data and move to device
                data = [data.to(device) for data in batch]
                input_data, y = data[:-1], data[-1]
                # pass through model and remove batch dim
                pred = model.predict_best(input_data, y)

                # get coordinates
                pred = helper.format_to_coordinates(pred.squeeze())
                y = helper.format_to_coordinates(y.squeeze())

                # get fde and ade
                displacements = metrics.calc_l2(pred, y)
                ade, fde = metrics.get_ade(
                    displacements), metrics.get_fde(displacements)

                # plot the vector map but dont display it until rest has been ploted
                plot_vector_map(scene_idx=scene_idx, data_dir=data_dir,
                                left_handed=left_handed, show=False)

                # plot prediction
                pred = np.lib.stride_tricks.sliding_window_view(
                    np.concatenate([[[0, 0]], pred.cpu()]), (2, 2)).reshape(-1, 4)
                for v in pred:
                    plt.quiver(v[0], v[1], v[2]-v[0], v[3]-v[1], angles='xy', scale_units='xy',
                               scale=1, headwidth=3, width=0.001, color='springgreen')

                # limit view
                limit = 70
                plt.xlim([-limit, limit])
                plt.ylim([-limit, limit])

                # Creating legend with color box and display plot
                gt = mpatches.Patch(color='green', label='ground truth future')
                focal_track = mpatches.Patch(
                    color='orange', label='ego agent past')
                agents = mpatches.Patch(color='blue', label='agents past')
                lanes = mpatches.Patch(color='black', label='lanes')
                prediction = mpatches.Patch(
                    color='springgreen', label=f'prediction: ADE {ade:.3f}, FDE {fde:.3f}')
                plt.legend(handles=[gt, focal_track,
                           agents, lanes, prediction])

                plt.savefig(f"../results/predictions/img{int(time.time())}.png")
                # plt.show()

        # if no scene found after loop print message
        if not scene_found:
            print('Scene not found. scene_idx might be out of range.')
