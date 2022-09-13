from pcl_pangu.context import set_context
from pcl_pangu.model import alpha, evolution, mPangu
import argparse
import pcl_pangu.model.panguAlpha_mindspore.inference_alpha_ms13 as inference_alpha_ms13

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='obs://pangu/models/26b/checkpiont_file/',
                    help="the path of model path")
parser.add_argument("--input_str", type=str, default='四川的省会是哪里',
                    help="question")
parser.add_argument("--model", type=str, default='2B6',
                    help="model")
parser.add_argument("--mindir_path", type=str, default='obs://pangu/models/26b/output/result.txt',
                    help="mindir path")
parser.add_argument("--output_file", type=str, default='obs://pangu/models/26b/output/',
                    help="output_file path")

args = parser.parse_args()
set_context(backend='mindspore')

config = alpha.model_config_npu(model=args.model,
                                load=args.model_path,
                                mindir_path=args.mindir_path)

alpha.inference(config,
                input=args.input_str,
                output_file=args.output_file,
                oneCardInference=True)
