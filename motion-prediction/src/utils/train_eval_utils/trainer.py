import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.utils.train_eval_utils.evaluater import evaluater, eval_batch


def trainer(model, device, log_dir, data_loader, val_loader, lr, epochs, weight_decay):

    # Optimizer and LR scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler =  optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(data_loader), epochs=epochs)

    # init writer for tensorboard'
    if log_dir:
        os.mkdir(log_dir)
        writer = SummaryWriter(log_dir)

    # start training
    for epoch in range(epochs):

        running_loss = 0.0  # reset running train loss
        running_ade = 0.0   # reset train ade

        model.train()
        for i, batch in enumerate(tqdm(data_loader, desc="training on batches")):
            
            # get data and move them to device
            data = [data.to(device) for data in batch]
            input_data, y = data[:-1], data[-1]

            # make optimization step
            optimizer.zero_grad()
            pred, loss = model.predict_best(input_data, y, return_loss=True)
            loss.backward()
            optimizer.step()

            # update learningrate
            scheduler.step()

            # calc metric
            with torch.no_grad():
                batch_ade, _, _, _ = eval_batch(pred, y)

            # add to running metrics    
            running_loss += loss.item()    
            running_ade += batch_ade
                
        # get metrics for train and val set
        train_loss = running_loss/(i+1)
        train_ade = running_ade/(i+1)
        val_loss, val_ade, _, _, _ = evaluater(model, device, val_loader)

        # write loss and metrics to tensorboard
        if log_dir:
            writer.add_scalars('Metric (ADE)', {'train': train_ade, 'val': val_ade}, epoch)
            writer.add_scalars('Loss (MSE)', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        # print train/val loss and metrics
        print(f"Epoch {epoch+1}:\tlast lr = {scheduler.get_last_lr()[0]:.3e}\ttrain loss = {train_loss:.3f} | val loss = {val_loss:.3f}\t train ADE = {train_ade:.3f} | val ADE = {val_ade:.3f}")
