from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import cv2
import os

class MyDataset(Dataset):
    def __init__(self, imgdir, transform=None):
        super(MyDataset, self).__init__()
        self.imgdir = imgdir
        self.transform = transform

        train_datas = []
        for sublabel, subdir in enumerate(os.listdir(self.imgdir)):
            for imgname in os.listdir(os.path.join(self.imgdir, subdir)):
                imgpath = os.path.join(self.imgdir, subdir, imgname)
                train_datas.append([imgpath, sublabel])
        self.train_datas = train_datas
        np.random.shuffle(self.train_datas)

    def __len__(self):
        return len(self.train_datas)

    def __getitem__(self, index):
        imgpath = self.train_datas[index][0]
        label = self.train_datas[index][1]
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        return img, label




if __name__ == "__main__":
    imgdir1 = '/media/txtx/f19f1c88-52d1-4fee-9746-dd97d2c44beb/data/cat_vs_dog/kaggle/1'
    class_names = os.listdir(imgdir1)
    transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ColorJitter(0.3),
                                    # transforms.ToTensor(),
                                    # transforms.Normalize((0.4914, 0.4822, 0.4465),(0.229, 0.224, 0.225)),
                                   ])
    my_dataset = MyDataset(imgdir1, transform)
    for index in range(len(my_dataset)):
        img, label = my_dataset.__getitem__(index)
        img = np.array(img)
        print(img.shape)
        print(label)
        cv2.putText(img, class_names[label], (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('1',img[:,:,::-1])
        cv2.waitKey()
        # print(2)
