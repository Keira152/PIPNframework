train:
python train.py -e 800 -b 3

evaluation:
python predict.py -c checkpoint.pth -m test
