
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


from MiDaS.depth_loss import compute_depth_loss
from yolo_bbox_decoder.yolo_loss import compute_loss

# garbage collection
import gc

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    # print('Apex recommended for mixed precision and faster training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

from main_model import OpNet


from torch.optim import lr_scheduler


def train(model, train_loader, device, yolo_args, midas_args, loss_weightage, model_save_path='./'):

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    model.train()

    # for i,(plane_data,yolo_data,depth_data) in enumerate(train_loader):
    for i,sample in enumerate(train_loader):

        batch_number+=1

        # sample = {'image': img, 'label': labels}
        # sample = totensor_object(sample)
        # sample["image_path"] = img_path
        # sample["origin_size"] = str([ori_w, ori_h])
        # sample["depth"] = depth_data

        dp_img_size,depth_img,depth_target = sample["origin_size"] , sample["image"] , sample["depth"]
        imgs, targets, paths= sample["image"], samole['lables'], sample["image_path"]

        # dp_img_size,depth_img,depth_target = depth_data
        # imgs, targets, paths, _ = yolo_data

        plane_out,yolo_out,midas_out = model.forward(yolo_inp,midas_inp,plane_inp)

        depth_loss = compute_depth_loss(midas_out, depth_target, dp_img_size)
        yolo_loss, yolo_loss_items = compute_loss(yolo_out, targets, model)


        all_loss =  (add_yolo_loss * yolo_loss) + (add_midas_loss * depth_loss)  # (add_plane_loss * plane_loss) +
        #all_loss = (add_yolo_loss * yolo_loss) + (add_midas_loss * ssim_out)
        # print('plane_loss : ', plane_loss)
        # print('yolo_loss : ', yolo_loss)
        # print('ssim_out : ', depth_loss)
        # print('all_loss :',all_loss)

        #optimizer.zero_grad()

        # Compute gradient
        if mixed_precision:
            with amp.scale_loss(all_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            all_loss.backward()
            pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gc.collect()
        # train_loader.dataset.data / train_loader.batch_size

        if batch_number%100 == 0:
            print('ALL LOSS : ', all_loss)








def run_model(model, train_loader, test_loader, epochs, device, yolo_args, midas_args, loss_weightage, model_save_path='./'):

    ## Model RUN!
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:')
        train(model,train_loader, criterion, optimizer, device, yolo_args, midas_args, loss_weightage, model_save_path='./')

    return model,train_trackers,test_trackers,incorrect_samples
