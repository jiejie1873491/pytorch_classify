import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

class_names = ['cat', 'dog', 'car']
model = models.resnet101()
model.fc = nn.Linear(model.fc.in_features, 3, bias=False)
model.load_state_dict(torch.load('model_resnet101/215.pth'))
model.to('cuda')
model.eval()

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225))])
with torch.no_grad():
    for imgname in glob.glob('/media/txtx/f19f1c88-52d1-4fee-9746-dd97d2c44beb/data/cat_vs_dog/kaggle/google/*.jpg'):
        img = cv2.imread(imgname)
        tmp_img = np.copy(img)
        # tmp_img = cv2.resize(tmp_img, (224, 224))
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        tmp_img = transform(tmp_img)
        tmp_img = tmp_img.unsqueeze(0)
        tmp_img = tmp_img.to('cuda')

        output = model(tmp_img)
        score = F.softmax(output)
        # s1 = F.sigmoid(output)
        pred = score.max(1, keepdim=True)[1]
        label = class_names[pred.item()]
        cv2.putText(img, '{}: {:.4f}'.format(label, score[0][pred].item()), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('1',img)
        cv2.waitKey()










