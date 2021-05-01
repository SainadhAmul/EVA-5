
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from model import Net

from depth_loss import compute_depth_loss
from yolo_loss import compute_loss

# garbage collection
import gc

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    # print('Apex recommended for mixed precision and faster training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

from main_model import OpNet

# temp = TSAIDataset("/content/updated_final_data/data/customdata/train.txt",
#                                                       (416, 416),
#                                                       is_training=True,data_dir_path = "/content/updated_final_data/data/customdata/images/")


# dataloader_temp = torch.utils.data.DataLoader(TSAIDataset("/content/updated_final_data/data/customdata/train.txt",
#                                                       (416, 416),
#                                                       is_training=True,data_dir_path = "/content/updated_final_data/data/customdata/images/"),
#                                           batch_size=32,
#                                           shuffle=True, num_workers=4, pin_memory=True)

def train(model, train_loader, criterion, optimizer, device, l1_factor =0,  **trackers):

    model = OpNet(yolo_cfg=cfg,midas_cfg=None,planercnn_cfg=config,path=midas_args.weights).to(device)

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







def test(model, test_loader, criterion, device, incorrect_samples, **trackers):

    test_losses = []

    model.eval()
    with torch.no_grad():

        correct_classified = 0
        for batch_number , (x_test,y_test) in enumerate(test_loader):

            x_test,y_test = x_test.to(device), y_test.to(device)
            pred = model.forward(x_test)
            loss = criterion(pred,y_test)
            test_losses.append(loss)

            correct_classified += (torch.max(pred,1)[1] == y_test).sum()

            ## INCORRECT PRED SAMPLES !
            output = pred.argmax(dim=1, keepdim=True)
            result = output.eq(y_test.view_as(output))

            if len(incorrect_samples) < 25:
                for i in range(test_loader.batch_size):
                    if not list(result)[i]:
                        incorrect_samples.append({
                            'prediction': list(output)[i],
                            'label': list(y_test.view_as(output))[i],
                            'image': list(x_test)[i]
                        })

        avg_loss = torch.mean(torch.tensor(test_losses))
        acc = round(correct_classified.item()/len(test_loader.dataset.data),5)

        prev_acc = trackers['test_acc']
        trackers['test_acc'] = prev_acc.append(acc)

        prev_losses = trackers['test_losses']
        trackers['test_losses'] = prev_losses.append(avg_loss.item())

        print('(TEST) Correct_classified : ' , correct_classified.item() ,' of 10000')
        print(f'(TEST) Loss : {avg_loss:4.4} Acc : {acc:4.5}')
        print('\n','*'*60 , '\n')




from torch.optim import lr_scheduler


def run_model(model, train_loader, test_loader, epochs, device, learning_rate, **regularization):

    # model = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    l2_factor = regularization['l2_factor']
    l1_factor = regularization['l1_factor']

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_factor)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.15)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    # ( factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

    ## TRACKERS
    train_losses = []
    train_acc = []
    train_trackers = {'train_acc':train_acc,'train_losses':train_losses}

    test_acc = []
    test_losses = []
    test_trackers = {'test_acc':test_acc,'test_losses':test_losses}

    incorrect_samples = []

    ## Model RUN!
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:')
        train(model,train_loader, criterion, optimizer, device,l1_factor =l1_factor, **train_trackers)
        # scheduler.step()
        test(model, test_loader, criterion, device, incorrect_samples, **test_trackers)
        scheduler.step(test_trackers['test_losses'][-1])

    return model,train_trackers,test_trackers,incorrect_samples
