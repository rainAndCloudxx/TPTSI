import argparse
from configs.default import get_config
import sys
import os
from data_process import Data_loader
from train_val import set_random_seeds,train_model,MetricCal
from model import Model,CombinedLoss
import torch
def parse_option():
    parser = argparse.ArgumentParser(description='Train on VUA All dataset')
    parser.add_argument('--cfg', type=str, default='./configs/config.yaml', metavar="FILE",
                        help='path to config file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--log', default='vua_all', type=str)
    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_random_seeds(args.seed)
    processor = Data_loader(args)

    model = Model(args=args)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.TRAIN.lr)

    if args.eval:
        test_loader = processor.get_test_data()
        MetricCal(args,model,test_loader,criterion)
    else:

        train_loader = processor.get_train_data()
        test_loader = processor.get_test_data()
        val_loader = processor.get_val_data()

        train_model(args,model, train_loader, val_loader, optimizer, criterion)
        MetricCal(args,model,test_loader,criterion)
if __name__ == '__main__':
    args = parse_option()
    main(args)