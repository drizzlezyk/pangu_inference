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
parser.add_argument("--model", type=str, default='2B6',
                    help="model")

args = parser.parse_args()
model_path = args.model_path
set_context(backend='mindspore')
config = alpha.model_config_npu(model=args.model, load=model_path)
alpha.inference(config, input=args.input_str)

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


class InferService():
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

    def load_inference(self, input_text):
        set_context(backend='mindspore')
        config = alpha.model_config_npu(model=self.model, load=self.model_path)
        alpha.inference(config, input=input_text, output_file='./output.txt')
        f = open('./output.txt', 'r', encoding='utf-8')
        output = f.readlines()
        return output
