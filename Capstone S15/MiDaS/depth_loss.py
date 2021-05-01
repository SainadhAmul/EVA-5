

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np
# for i,(plane_data,yolo_data,depth_data) in pbar:

# dp_img_size,depth_img,depth_target = depth_data

# plane_out,yolo_out,midas_out = model.forward(yolo_inp,midas_inp,plane_inp)

def compute_depth_loss(midas_out, depth_target, dp_img_size):

        dp_prediction = midas_out

        dp_prediction = (
                        torch.nn.functional.interpolate(
                            dp_prediction.unsqueeze(1),
                            size=tuple(dp_img_size[:2]),
                            mode="bicubic",
                            align_corners=False,
                        )
                        #.unsqueeze(0)
                        #.cpu()
                        #.numpy()
                        )
        bits=2
        depth_min = dp_prediction.min()
        depth_max = dp_prediction.max()

        max_val = (2**(8*bits))-1

        if depth_max - depth_min > np.finfo("float").eps:
            depth_out = max_val * (dp_prediction - depth_min) / (depth_max - depth_min)
        else:
            depth_out = 0

        depth_target = torch.from_numpy(np.asarray(depth_target)).to(device).type(torch.cuda.FloatTensor).unsqueeze(0)
        #print('depth_target',depth_target.size())

        depth_target = (
                        torch.nn.functional.interpolate(
                            depth_target.unsqueeze(1),
                            size=dp_img_size[:2],
                            mode="bicubic",
                            align_corners=False
                        )
                        #.unsqueeze(0)
                        #.cpu()
                        #.numpy()
                        )


        depth_pred = Variable( depth_out,  requires_grad=True)
        depth_target = Variable( depth_target, requires_grad = False)


        ssim_loss = pytorch_ssim.SSIM() #https://github.com/Po-Hsun-Su/pytorch-ssim
        #print('ssim_loss :',ssim_loss(depth_pred,depth_target))
        #print('msssim :',msssim(depth_pred,depth_target))
        ssim_out = torch.clamp(1-ssim_loss(depth_pred,depth_target),min=0,max=1) #https://github.com/jorge-pessoa/pytorch-msssim

        loss_fn = nn.MSELoss()
        RMSE_loss = torch.sqrt(loss_fn(depth_pred, depth_target))

        depth_loss = (0.0001*RMSE_loss) + ssim_out

        return depth_loss
