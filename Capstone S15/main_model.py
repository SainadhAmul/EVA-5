
import torch
import torch.nn as nn

from resnext_head import _resnext_backbone

from MiDaS.midas_decoder_de import MidasDecoder

from yolo_bbox_decoder.bbox_decoder import *
from yolo_bbox_decoder.bbox_decoder import Darknet
from yolo_bbox_decoder.utils import torch_utils



class OpNet(nn.Module):

	'''
		First Iteration: Network for detecting objects, generate depth map
	'''

	def __init__(self, yolo_cfg, midas_cfg):

		super(OpNet, self).__init__()

		"""
			Get required configuration for all the 3 models

		"""
		self.yolo_params = yolo_cfg
		self.midas_params = midas_cfg
		# self.planercnn_params = planercnn_cfg
		self.path = midas_cfg['weights']

		use_pretrained = False if self.path is None else True

		print('use_pretrained',use_pretrained)
		print('path',self.path)

		self.encoder = _resnext_backbone(use_pretrained)

		# self.plane_decoder = MaskRCNN(self.planercnn_params,self.encoder) #options, config, modelType='final'

        # DEPTH EST DECODER
		self.depth_decoder = MidasDecoder(self.midas_params['weights'])

        ## BBX REGR DECODER
		self.bbox_decoder =  Darknet(self.yolo_params)


		self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(1, 1), padding=0, bias=False)
		self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), padding=0, bias=False)
		self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding=0, bias=False)

		self.info(False)

	def forward(self, yolo_inp, midas_ip):

		x = yolo_inp
		#x = midas_ip
		#print('yolo_ip',yolo_ip.shape,yolo_ip[0][0][0][0])
		#print('midas_ip',midas_ip.shape,midas_ip[0][0][0][0])

		# Encoder blocks
		layer_1 = self.encoder.layer1(x)
		layer_2 = self.encoder.layer2(layer_1)
		layer_3 = self.encoder.layer3(layer_2)
		layer_4 = self.encoder.layer4(layer_3)

		#print('%'*30,'layer_1',layer_1[0][0][0][0])
		#print('layer_4',layer_4[0][0])

		Yolo_75 = self.conv1(layer_4)
		Yolo_61 = self.conv2(layer_3)
		Yolo_36 = self.conv3(layer_2)

		# if plane_ip is not None:
		# 	plane_ip['input'][0] = yolo_ip
		# 	# PlaneRCNN decoder
		# 	plane_out = self.plane_decoder.forward(plane_ip,[layer_1, layer_2, layer_3, layer_4])
		# else:
		# 	plane_out = None

		if midas_ip is not None:
			# MiDaS depth decoder
			depth_out = self.depth_decoder([layer_1, layer_2, layer_3, layer_4])
		else:
			depth_out = None

		#print('en Yolo_75 :',Yolo_75.shape)
		#print('en Yolo_61 :',Yolo_61.shape)
		#print('en Yolo_36 :',Yolo_36.shape)

		#print('^'*66,'Yolo_75',Yolo_75[0][0])


		#YOLOv3 bbox decoder
		if not self.training:
			inf_out, train_out = self.bbox_decoder(Yolo_75,Yolo_61,Yolo_36)
			bbox_out=[inf_out, train_out]
		else:
			bbox_out = self.bbox_decoder(Yolo_75,Yolo_61,Yolo_36)

		#print('depth_out',depth_out)

		return  bbox_out, depth_out
		# return  plane_out, bbox_out, depth_out

	def info(self, verbose=False):
		torch_utils.model_info(self, verbose)
