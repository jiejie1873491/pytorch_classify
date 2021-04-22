import matplotlib.pyplot as plt
import numpy as np
val_acc = []
with open('model_resnet18/val_log.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        data = line.strip().split()
        val_acc.append(float(data[-1]))

# plt.plot(val_acc[:])
# plt.show()

train_acc = []
with open('model_resnet18/train_log.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        data = line.strip().split()
        train_acc.append(float(data[-1]))

acc = np.concatenate((np.array(val_acc).unsqueeze(0),np.array(train_acc).unsqueeze(0)))
plt.plot(acc[:])
plt.show()