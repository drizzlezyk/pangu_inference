#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2022/7/20
# @Author: 2022 PCL
import os
import sys
from loguru import logger
from ..launcher_torch import launch

from pcl_pangu.context import check_context
from .config_alpha import DISTRUBUTED_CONFIG, model_config_gpu, model_config_npu


def train(config):
    print('---------------------------- train config ----------------------------')
    print("> Base Model: [alpha]")
    print("> Model Size: [{}]".format(config.model))
    print("> data_path: {}".format(config.data_path))
    print("> global batch_size: {}".format(config.batch_size))
    print("> save to path: {}".format(config.save))
    print('--------------------------- end of config ----------------------------')

    if check_context()=='pytorch':
        assert isinstance(config, model_config_gpu)
        script_args = config._get_training_script_args()
        py_script = '/panguAlpha_pytorch/pretrain_gpt2.py'
        run_pt(script_args, py_script)
    elif check_context()=='mindspore':
        assert isinstance(config, model_config_npu)
        config_dict = config._cover_modelarts_training_args()
        run_ms_train(config_dict)



def fine_tune(config):
    print('-------------------------- finetune config ---------------------------')
    print("> Base Model: [alpha]")
    print("> Model Size: [{}]".format(config.model))
    print("> data_path: {}".format(config.data_path))
    print("> global batch_size: {}".format(config.batch_size))
    print("> save to path: {}".format(config.save))
    print('--------------------------- end of config ----------------------------')

    if check_context()=='pytorch':
        from .config_alpha import DEFAULT_CONFIG
        DEFAULT_CONFIG['finetune'] = True
        script_args = config._get_training_script_args()
        py_script = '/panguAlpha_pytorch/pretrain_gpt2.py'
        run_pt(script_args, py_script)
    elif check_context()=='mindspore':
        assert isinstance(config, model_config_npu)
        config_dict = config._cover_modelarts_training_args()
        run_ms_train(config_dict)


def inference(config, top_k=1, top_p=0.9, input=None, input_file=None,
              generate_max_tokens=128, output_file=None, oneCardInference=True, mindir_path=''):
    assert generate_max_tokens > 0 and generate_max_tokens <= 800, "> generate_max_tokens always in (0, 800]"
    print('--------------------------- inference config --------------------------')
    print("> Base Model: [alpha]")
    print("> Model Size: [{}]".format(config.model))
    print("> global batch_size: {}".format(config.batch_size))
    print("> generate_max_tokens length: {}".format(generate_max_tokens))
    print('---------------------------- end of config ----------------------------')

    if check_context()=='pytorch':
        from .config_alpha import DEFAULT_CONFIG
        DEFAULT_CONFIG['finetune'] = True
        config.batch_size = 1
        script_args = config._get_training_script_args(oneCardInference=oneCardInference)
        py_script = '/panguAlpha_pytorch/tools/generate_samples_Pangu.py'
        script_args.append('--top-k={}'.format(top_k))
        script_args.append('--top-p={}'.format(top_p))
        if input is not None:
            script_args.append('--sample-input={}'.format(input.encode('utf-8').hex()))
        if input_file is not None:
            script_args.append('--sample-input-file={}'.format(input_file))
        if output_file is not None:
            script_args.append('--sample-output-file={}'.format(output_file))
        if generate_max_tokens is not None:
            script_args.append('--generate_max_tokens={}'.format(generate_max_tokens))
        run_pt(script_args, py_script)

    elif check_context()=='mindspore':
        assert isinstance(config, model_config_npu)
        # config_dict = config._cover_modelarts_training_args(oneCardInference=oneCardInference)
        # config_dict['top_k'] = top_k
        # config_dict['top_p'] = top_p
        # config_dict['input'] = input
        # config_dict['input_file'] = input_file
        # config_dict['output_file'] = output_file
        # config_dict['generate_max_tokens'] = generate_max_tokens
        # config_dict['mindir_path'] = mindir_path
        # print(config_dict.keys())

        result = run_ms_inference(config)
        return result

def run_pt(script_args, py_script, **kwargs):
    current_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(current_dir+'/panguAlpha_pytorch')
    py_script = current_dir + py_script
    logger.info("> Running {} with args: {}".format(py_script, script_args))
    launch(training_script=py_script,
           training_script_args=script_args,
           **DISTRUBUTED_CONFIG,
           **kwargs)


def run_ms_train(config_dict):
    current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    sys.path.append(current_dir + '/panguAlpha_mindspore')
    from pcl_pangu.model.panguAlpha_mindspore.train_alpha_ms13 import opt, setup_args, main, alpha_add_args
    new_opt = setup_args(opt, config_dict)
    new_opt = alpha_add_args(new_opt)
    main(new_opt)


def run_ms_inference(config_dict):
    # current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    # sys.path.append(current_dir + '/panguAlpha_mindspore')
    from pcl_pangu.model.panguAlpha_mindspore.inference_alpha_ms13 import load_inference
    # print(config_dict.keys())
    return load_inference(config_dict)

