from model import PINN
from dataset import myData

import torch
import torch.nn as nn

import argparse
import logging
import time
import random
import numpy as np
import pandas as pd
import os

def predict(net, device, test_dataset, mode='test'):
    net.eval()
    time_sum = 0
    n_test = len(test_dataset)
    acc_list = np.zeros((n_test, 1))
    if mode == 'test':
        logging.info("Start Testing:")
        for idx in range(n_test):
            label = np.array([test_dataset[idx]['label']])
            input1 = np.array([test_dataset[idx]['image']])
            assert type(input1) == np.ndarray, f"the input type is {type(input1)}"
            xyz = input1
            input2 = np.array([test_dataset[idx]['surface']])
            input1 = torch.from_numpy(input1).transpose(1,2).to(device)
            input2 = torch.from_numpy(input2).transpose(1,2).to(device)

            time_start = time.time()
            pred, _, _, _ = net(input1,input2)
            time_end = time.time()

            time_consuming = time_end-time_start
            logging.info(f"time consuming is {time_consuming}")
            time_sum = time_sum + time_consuming

            prediction = pred.cpu().detach().numpy().reshape(-1, 1)
            accuracy = np.mean(1-np.abs((prediction-label)/label))
            acc_list[idx:idx+1,0:1] = accuracy
        
        logging.info(f"the time consuming is {time_sum/n_test}")
        acc_list = pd.DataFrame(acc_list)
        acc_list.to_csv(f"{work_dir}accuracy.csv", index=False, header=False, float_format='%.3f')

    elif mode == 'predict':
        logging.info("Start Predicting:")
        idx = random.randint(0,len(test_dataset)-1)
        logging.info(f"Testing data idx is {idx+1}")

        label = np.array([test_dataset[idx]['label']])
        input1 = np.array([test_dataset[idx]['image']])
        assert type(input1) == np.ndarray, f"the input type is {type(input1)}"
        xyz = input1
        input2 = np.array([test_dataset[idx]['surface']])
        input1 = torch.from_numpy(input1).transpose(1,2).to(device)
        input2 = torch.from_numpy(input2).transpose(1,2).to(device)

        time_start = time.time()
        pred, _, _, _ = net(input1,input2)
        time_end = time.time()
        logging.info(f"the time consuming is {time_end-time_start}")

        prediction = pred.cpu().detach().numpy().reshape(-1, 1)
        res = np.zeros((len(prediction),5))
        res[:,0:3] = xyz
        res[:,3:4] = prediction
        res[:,4:5] = label

        res = pd.DataFrame(res)
        res.to_csv(f"{work_dir}test.csv", index=False, header=False, float_format='%.3f')
    else:
        raise ValueError("please determine the mode!")


def get_args():
    parser = argparse.ArgumentParser(description='Predict temperature field from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-tg', '--test-geometry-dir', type=str, default=None, dest='test_geo_dir')
    parser.add_argument('-ts', '--test-surface-dir', type=str, default=None, dest='test_sur_dir')
    parser.add_argument('-c', '--checkpoint', default='models/checkpoint.pth', metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('-m', '--mode',type=str, default='test')
    parser.add_argument('-w', '--work-dir',type=str, default='work_dir/', dest='work_dir')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Create working directory
    work_dir=args.work_dir
    try:
        os.mkdir(work_dir)
        logging.info('Created working directory')
    except OSError:
        pass

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename=f'{work_dir}testing.log')

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # instantiate the PINN model
    net = PINN()
    # Distribute training over GPUs
    net = nn.DataParallel(net)
    net.to(device=device)

    # Instantiate datasets
    path_dir1 = args.test_geo_dir if args.test_geo_dir is not None else 'dataset/valid_data1/'
    path_dir2 = args.test_sur_dir if args.test_sur_dir is not None else 'dataset/valid_data2/'
    logging.info(f"Loading dataset from {path_dir1} and {path_dir2}")
    test_dataset = myData(path_dir1, path_dir2)

    # Loading the trained model
    logging.info("Loading model {}".format(args.checkpoint))
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    logging.info("Model loaded !")

    # Strat the test
    predict(net, device, test_dataset, mode=args.mode)