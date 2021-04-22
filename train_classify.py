'''
1 loading data and process
2 define the network
3 define the loss and optimizer
4 train
5 test
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from my_datasets import MyDataset
import matplotlib.pyplot as plt


BACTH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_model_dir = 'model_resnet101'
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

# 1 loading data and process
train_dir = '/media/txtx/f19f1c88-52d1-4fee-9746-dd97d2c44beb/data/cat_vs_dog/kaggle/train_keras'
train_transform = transforms.Compose([
                                      transforms.ToPILImage(),
                                      transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(5),
                                      transforms.ColorJitter(0.3),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225))
                                     ])
train_dataset = MyDataset(imgdir=train_dir, transform=train_transform)
train_dataloader = DataLoader(train_dataset, BACTH_SIZE, shuffle=True)


val_dir = '/media/txtx/f19f1c88-52d1-4fee-9746-dd97d2c44beb/data/cat_vs_dog/kaggle/test_keras'
val_transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225))
                                    ])
val_dataset = MyDataset(imgdir=val_dir, transform=val_transform)
val_dataloader = DataLoader(val_dataset, BACTH_SIZE, shuffle=True)
class_names = os.listdir(val_dir)
# for val_images, val_labels in val_dataloader:
#     val_images = val_images.to(DEVICE)
#     val_labels = val_labels.to(DEVICE)
#     pass


# 2 define the network
model = models.resnet101(pretrained=True)
for param in model.parameters():
    param.requires_grad = True

fc_inputs = model.fc.in_features
model.fc = nn.Linear(fc_inputs, len(os.listdir(train_dir)), bias=False)
# for param in model.parameters():
#     print(param.requires_grad)

use_pretrain = True
if use_pretrain:
    pretrain_model = '32.pth'
    model.load_state_dict(torch.load(os.path.join(save_model_dir, pretrain_model)))
    start_epoch = int(pretrain_model.split('.')[0]) + 1
else:
    start_epoch = 1
model = model.to(DEVICE)

wirter = SummaryWriter(save_model_dir)
# 3 define the loss and optimizer
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters())
lr = optimizer.state_dict()['param_groups'][0]


# 4 train the network
def train(train_dataloader, model, loss_fun, optimizer, epoches):
    print('Start to train...')
    for epoch in range(start_epoch, epoches+1):
        model.train()
        one_epoch_train_loss = []
        one_epoch_train_acc = []
        for batch_idx ,batches_data in enumerate(train_dataloader):
            images = batches_data[0].to(DEVICE)
            labels = batches_data[1].to(DEVICE)

            optimizer.zero_grad()
            output = model(images)
            loss = loss_fun(output, labels)
            loss.backward()
            optimizer.step()

            one_epoch_train_loss.append(loss.item())
            pred = output.max(1, keepdim=True)[1]
            acc = pred.eq(labels.view_as(pred)).sum().item() / len(images)
            one_epoch_train_acc.append(acc)
            wirter.add_scalar('train_loss', loss.item(), (epoch-1) * len(train_dataloader) + batch_idx)

            print('Train Epoch: {}/{}, {} [{}/{} ({:.0f}%)]    Loss: {:.6f},  Acc: {:.4f}'.format(
                epoch, epoches, batch_idx, (batch_idx+1) * len(images), len(train_dataloader.dataset),
                100. * (batch_idx+1) / len(train_dataloader), loss.item(), acc))

        img_grid = torchvision.utils.make_grid(batches_data[0])
        wirter.add_image('train batch images', img_grid)
        print('Train Epoch: {}, meanLoss: {:.6f},  meanAcc: {:.4f}'.format(epoch, np.mean(one_epoch_train_loss), np.mean(one_epoch_train_acc)))
        with open(os.path.join(save_model_dir, 'train_log.txt'), 'a') as f:
            f.write('Train Epoch: {}, meanLoss: {:.6f},  meanAcc: {:.4f}\n'.format(epoch, np.mean(one_epoch_train_loss), np.mean(one_epoch_train_acc)))
        
        if epoch%2==0:
            torch.save(model.state_dict(),  os.path.join(save_model_dir, str(epoch) + '.pth'))

        with torch.no_grad():
            model.eval()
            one_epoch_val_loss = []
            one_epoch_val_acc = []
            for val_batch_idx, val_batches_data in enumerate(val_dataloader):
                val_images = val_batches_data[0].to(DEVICE)
                val_labels = val_batches_data[1].to(DEVICE)

                val_output = model(val_images)
                val_loss = loss_fun(val_output, val_labels)
                one_epoch_val_loss.append(val_loss.item())
                val_pred = val_output.max(1, keepdim=True)[1]
                val_acc = val_pred.eq(val_labels.view_as(val_pred)).sum().item() / len(val_images)
                one_epoch_val_acc.append(val_acc)
                wirter.add_scalar('val_loss', val_loss.item(), (epoch-1) * len(val_dataloader) + val_batch_idx)

            img_grid = torchvision.utils.make_grid(val_batches_data[0])
            wirter.add_image('val batch images', img_grid)

            aa = np.zeros((len(val_images),224,224,3), np.uint8)
            for i in range(len(val_images)):
                tmp_img = val_images[i].cpu().numpy()
                tmp_img = np.transpose(tmp_img, [1,2,0])
                tmp_img = (tmp_img *np.array([0.229, 0.224, 0.225]) + np.array([0.4914, 0.4822, 0.4465]))*255
                tmp_img = tmp_img.astype(np.uint8)
                tmp_img1 = tmp_img.copy()
                val_pred1 = val_pred.cpu().numpy()
                cv2.putText(tmp_img1, class_names[val_pred1[i][0]], (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1) 
                # cv2.imshow('', tmp_img1[:, :, ::-1]); cv2.waitKey();cv2.destroyAllWindows()
                aa[i] = tmp_img1

            bb = torch.from_numpy(np.transpose(aa, [0,3,1,2]))
            img_grid1 = torchvision.utils.make_grid(bb)
            wirter.add_image('predict images', img_grid1)

            print('val Epoch: {}, meanLoss: {:.6f},  meanAcc: {:.4f}\n'.format(epoch, np.mean(one_epoch_val_loss), np.mean(one_epoch_val_acc)))    
            with open(os.path.join(save_model_dir, 'val_log.txt'), 'a') as f:
                f.write('val Epoch: {}, meanLoss: {:.6f},  meanAcc: {:.4f}\n'.format(epoch, np.mean(one_epoch_val_loss), np.mean(one_epoch_val_acc)))    


if __name__ == "__main__":
    train(train_dataloader, model, loss_fun, optimizer, 500)