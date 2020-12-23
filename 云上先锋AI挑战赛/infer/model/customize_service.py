# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from model_service.pytorch_model_service import PTServingBaseService

import time
from metric.metrics_manager import MetricsManager
import log
from io import BytesIO
import base64
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000000000
logger = log.getLogger(__name__)

import cv2, sys

sys.path.append('.')
from mmseg.apis import inference_segmentor, init_segmentor


class ModelClass(PTServingBaseService):
    def __init__(self, model_name='', model_path=r'./best_model.pth'):
        self.model_name = model_name
        self.model_path = model_path

        config_file = '/home/mind/model/config.py'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = init_segmentor(config_file, self.model_path, device=self.device)
         

    def _preprocess(self, data):
        """
        本函数无需修改
        """
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                
                
                preprocessed_data[k] = img
       
        return preprocessed_data

    def _inference(self, data):
        src_img = data['images']  # 本行代码必须保留，且无需修改
       
        output = inference_segmentor(self.model, src_img)[0]
        pred_ = []
        pred_.append(output.tolist())

        result = {
            'pred': [],
        }

        result['pred'].append(pred_)

        return result
   
