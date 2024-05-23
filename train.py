import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import PINN
from dataset import myData
from metrics import PIPNloss, pointnetloss, PDEloss, dataloss

def train_net(net, 
              device,
              train_dataset,
              valid_dataset,
              dir_checkpoint,
              epochs=800,
              batch_size=3,
              save=True):
    
    n_train = len(train_dataset)
    n_val = len(valid_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Prepares the summary file
    writer = SummaryWriter(comment=f'EPOCHS_{epochs}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save}
        Device:          {device.type}
    ''')

    # Train loop
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        train_error = 0
        batch = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='pcd') as pbar:
            for data in train_loader:
                batch = batch + 1
                input1 = data['image'].to(device, dtype=torch.float32)
                input2 = data['surface'].to(device, dtype=torch.float32)
                label = data['label'].to(device, dtype=torch.float32)
                tem_pred, matrix3x3, matrix4x4, matrix64x64 = net(input1.transpose(1,2), input2.transpose(1,2))
                assert tem_pred.size()[1] == 10000, f"tem_pred shape is {tem_pred.shape}"

                # Compute loss
                loss = PIPNloss(tem_pred, matrix3x3, matrix4x4, matrix64x64, input1, label)
                loss1 = dataloss(tem_pred, label)
                loss2 = PDEloss(tem_pred, input1)
                loss3 = pointnetloss(tem_pred, matrix3x3, matrix4x4, matrix64x64)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (iteration)': loss.item(), 'dataloss': loss1.item(), 'PDEloss': loss2.item(), 'matrixloss': loss3.item()})

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(1)

                error = (torch.sum(torch.abs(tem_pred.sub(label)).div(label))).div(label.size()[0]*label.size()[1])
                train_error += error
                
        train_error = train_error / batch
        global_step += 1
        writer.add_scalar('Loss/train', epoch_loss, global_step)
        writer.add_scalar('Error/train', train_error, global_step)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        scheduler.step(train_error)

        # validation
        net.eval()
        valid_error = 0
        batch = 0
        with tqdm(total=n_val, desc='Validation round', unit='pcd') as pbar:
            with torch.no_grad():
                for data in val_loader:
                    batch = batch + 1
                    input1 = data['image'].to(device, dtype=torch.float32)
                    input2 = data['surface'].to(device, dtype=torch.float32)
                    label = data['label'].to(device, dtype=torch.float32)
                    tem_pred, _, _, _ = net(input1.transpose(1,2), input2.transpose(1,2))
                    error = (torch.sum(torch.abs(tem_pred.sub(label)).div(label))).div(label.size()[0]*label.size()[1])
                    valid_error += error

                    pbar.update(1)

        valid_error = valid_error / batch
        writer.add_scalar('Error/validation', valid_error, global_step)
        valid_error_numpy = valid_error.cpu().detach().numpy()

        # save the model
        if save:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if valid_error_numpy < min_error:
                min_error = valid_error_numpy
                torch.save(net.state_dict(), 
                           dir_checkpoint + f'error{valid_error_numpy}_epoch{epoch + 1}.pth')
            if epoch == epochs-1:
                torch.save(net.state_dict(), 
                           dir_checkpoint + f'error{valid_error_numpy}_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='PIPointnet train script')
    parser.add_argument('-tg', '--training-geometry-dir', type=str, default=None, dest='train_geo_dir')
    parser.add_argument('-ts', '--training-surface-dir', type=str, default=None, dest='train_sur_dir')
    parser.add_argument('-vg', '--validation-geometry-dir', type=str, default=None, dest='val_geo_dir')
    parser.add_argument('-vs', '--validation-surface-dir', type=str, default=None, dest='val_sur_dir')
    parser.add_argument('-e', '--epochs', type=int, default=800)
    parser.add_argument('-b', '--batch-size', type=int, default=3, dset='batch_size')
    parser.add_argument('-c', '--dir_checkpoint', type=str, default='checkpoints/')
    parser.add_argument('-l', '--load', type=str, default=False, help='Load model from a .pth file', dest='load')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename='training.log')
    args = get_args()

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # instantiate the PINN model
    net = PINN()

    # Distribute training over GPUs
    net = nn.DataParallel(net)

    # Load weights from file
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    # Determine the optimizer and scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)#定义优化器的学习速率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20)

    # Instantiate datasets
    path_dir1 = args.train_geo_dir if args.train_geo_dir is not None else 'train_data1/'
    path_dir2 = args.train_sur_dir if args.train_sur_dir is not None else 'train_data2/'
    path_dir3 = args.val_geo_dir if args.val_geo_dir is not None else 'valid_data1/'
    path_dir4 = args.val_sur_dir if args.val_sur_dir is not None else 'valid_data2/'

    train_dataset = myData(path_dir1, path_dir2, transform=None)
    valid_dataset = myData(path_dir3, path_dir4, transform=None)


    try:
        train_net(net=net, 
                  device=device,
                  train_dataset=train_dataset,
                  valid_dataset=valid_dataset,
                  dir_checkpoint=args.dir_checkpoint,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  save=True)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)