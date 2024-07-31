
import argparse

def get_parser():

        parser = argparse.ArgumentParser(description='Model')
        parser.add_argument('--sequence-length', '-l', type=int, default=100,
                            help='length of each input sequence')
        parser.add_argument('--epochs', type=int, metavar='N',
                            help='number of total epochs to run (default: 90)')
        return parser