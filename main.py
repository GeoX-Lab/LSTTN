import argparse
from easytorch import launch_training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str)
    parser.add_argument("--gpus", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch_training(args)
