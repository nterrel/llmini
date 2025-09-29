import argparse
import logging


def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tiny", "complex"], default="tiny",
                        help="Choose the model architecture: 'tiny' or 'complex'")
    return parser.parse_args()
