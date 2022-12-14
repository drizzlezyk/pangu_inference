# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from pcl_pangu.context import set_context
from pcl_pangu.model import alpha, evolution, mPangu
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='./model',
                    help="the path of model path")
parser.add_argument("--input_str", type=str, default='四川的省会是哪里？',
                    help="question")
parser.add_argument("--output_file", type=str, default='../output.txt',
                    help="output file")

args = parser.parse_args()
model_path = args.model_path
set_context(backend='pytorch')
config = alpha.model_config_gpu(model='350M', load=model_path)
alpha.inference(config, input=args.input_str, output_file=args.output_file)

# from pcl_pangu.context import set_context
# from pcl_pangu.dataset import txt2mindrecord
# from pcl_pangu.model import alpha, evolution, mPangu
#
# set_context(backend='mindspore')
# data_path = 'path/of/training/dataset'
# txt2mindrecord(input_glob='your/txt/path/*.txt', output_prefix=data_path)
# config = alpha.model_config_npu(model='350M',
#                                 load='path/to/save/ckpt',
#                                 data_path=data_path)
# alpha.train(config)