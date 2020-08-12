import sys
import argparse

import torch

from pixelssl import ssl_algorithm
from pixelssl.nn import optimizer, lrer
from pixelssl.nn.func import pytorch_support
from pixelssl.utils import logger, cmd


def create_parser(algorithm):
    parser = argparse.ArgumentParser(description='PixelSSL Static Script Parser')

    if not algorithm in ssl_algorithm.SSL_ALGORITHMS:
        logger.log_err('Unknown semi-supervised learning algorithm: {0}\n'
                       'The support algorithms are: {1}\n'
                       .format(algorithm, ssl_algorithm.SSL_ALGORITHMS))

    optimizer.add_parser_arguments(parser)
    lrer.add_parser_arguments(parser)
    ssl_algorithm.__dict__[algorithm].add_parser_arguments(parser)
    
    return parser


def run_script(config, proxy_file, proxy_class):
    # PixelSSL requires PyTorch >= 1.0.0
    pytorch_support(required_version='1.0.0', info_str='PixelSSL')

    # help information
    if len(sys.argv) > 1 and sys.argv[1] in ['help', '--help', 'h', '-h']:
        config['h'] = True

    # create parser and parse args from config
    parser = create_parser(config['ssl_algorithm'])
    proxy_file.add_parser_arguments(parser)
    args = cmd.parse_args(parser, config)

    task_proxy = proxy_class(args)
    task_proxy.run()
