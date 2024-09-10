
import argparse

def get_parser():

        parser = argparse.ArgumentParser(description='Model')
        parser.add_argument('--model', type=str)
        parser.add_argument('--val-split', type=float, default=0.1)
        parser.add_argument('--sequence-length', '-l', type=int, default=100,
                            help='length of each input sequence')
        parser.add_argument('--batch-size', type=int, default=256,
                            help='batch size')
        parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4,
                            help='learning rate')
        parser.add_argument('--weight-decay', '-wd', type=float, default=0.004,
                            help='weight decay')
        parser.add_argument('--epochs', type=int, metavar='N',
                            help='number of total epochs to run (default: 90)')
        parser.add_argument('--dataset', type=str,
                            help='dataset profile source path')
        parser.add_argument('--checkpoint', type=str, default=None,
                            help='checkpoint')
        parser.add_argument('--monitor', type=str, default="val_loss")
        parser.add_argument('--monitor-mode', type=str, default="min")
        return parser