from flask import Flask, request
import json
from pcl_pangu.context import set_context
from pcl_pangu.model import alpha, evolution, mPangu

APP = Flask(__name__)
MODEL_PATH = '../'
OUT_PATH = ''

@APP.route('./test', method=['GET'])
def test_func():
    return json.dumps({'test':'true'})


@APP.route('infer/text/', method=['POST'])
def inference_text():

    input = request.files['file']
    set_context(backend='mindspore')
    config = alpha.model_config_npu(model='350M', load=MODEL_PATH)

    alpha.inference(config, input=input)

