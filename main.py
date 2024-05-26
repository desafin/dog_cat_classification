import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
from copy import deepcopy
import time

import random
import os

# device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (device)

seed = 42 # seed 값 설정
random.seed(seed) # 파이썬 난수 생성기
os.environ['PYTHONHASHSEED'] = str(seed) # 해시 시크릿값 고정
np.random.seed(seed) # 넘파이 난수 생성기

torch.manual_seed(seed) # 파이토치 CPU 난수 생성기
torch.backends.cudnn.deterministic = True # 확정적 연산 사용 설정
torch.backends.cudnn.benchmark = False   # 벤치마크 기능 사용 해제
torch.backends.cudnn.enabled = False        # cudnn 기능 사용 해제

if device == 'cuda':
    torch.cuda.manual_seed(seed) # 파이토치 GPU 난수 생성기
    torch.cuda.manual_seed_all(seed) # 파이토치 멀티 GPU 난수 생성기

import glob
from sklearn.model_selection import train_test_split

train_dir = 'data/train'
test_dir = 'data/test1'

all_train_files = glob.glob(os.path.join(train_dir, '*.jpg'))

test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
train_labels = [path.split('/')[-1].split('.')[0] for path in all_train_files]
train_list, val_list = train_test_split(all_train_files, test_size = 0.1, stratify = train_labels, random_state=seed)
print (len(train_list), len(val_list))

print (train_list[0])

# 파일 이름을 출력하여 확인
def check_file_list(file_list):
    for file_path in file_list:
        print(f'Processing file: {file_path}')
        label_str = os.path.basename(file_path).split('.')[0]
        if label_str not in ['dog', 'cat']:
            print(f'Warning: Unexpected label {label_str} in file {file_path}')

# train_list와 val_list의 파일 이름을 확인
check_file_list(train_list)
check_file_list(val_list)

from torchvision import transforms

input_size = 224
transforms_for_train = transforms.Compose([
    transforms.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transforms_for_val_test = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class CustomDataset(Dataset):
    def __init__(self, file_list, transform=None, is_test=False):
        self.file_list = file_list
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img_transform = self.transform(img)

        if self.is_test:
            return img_transform, os.path.basename(img_path)

        label = os.path.basename(img_path).split('.')[0]
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0
        else:
            raise ValueError(f'Unknown label in file {img_path}')

        return img_transform, label


dataset_train = CustomDataset(train_list, transform=transforms_for_train)
dataset_valid = CustomDataset(val_list, transform=transforms_for_val_test)
dataset_test = CustomDataset(test_list, transform=transforms_for_val_test, is_test=True)

from torch.utils.data import DataLoader  # 데이터 로더 클래스

train_batches = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
val_batches = DataLoader(dataset=dataset_valid, batch_size=64, shuffle=False)
test_batches = DataLoader(dataset=dataset_test, batch_size=64, shuffle=False)


import timm


model = timm.create_model("vit_base_patch32_224_in21k", pretrained=True)
model.head = nn.Sequential(
    nn.Linear(768, 21843, bias=True),
    nn.LeakyReLU(),
    nn.BatchNorm1d(21843),
    nn.Linear(21843, 512, bias=True),
    nn.LeakyReLU(),
    nn.BatchNorm1d(512),
    nn.Linear(512, 1, bias=True),
    nn.Sigmoid()
)
model.to(device)
loss_func = nn.BCELoss()
# optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.001)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0001)
# optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# learning rate scheduler
# https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps = 0,
  num_training_steps = 10
)


def train_model(model, criterion, optimizer, early_stop, epochs, train_loader, valid_loader):
    train_losses, train_accuracies, valid_losses, valid_accuracies, lowest_loss, lowest_epoch = list(), list(), list(), list(), np.inf, 0

    # DEBUG
    progress_count = 0

    for epoch in range(epochs):
        train_loss, train_accuracy, train_corrects, valid_loss, valid_accuracy, valid_corrects = 0, 0, 0, 0, 0, 0
        train_correct, valid_correct = 0, 0

        start = time.time()
        model.train()
        for train_x, train_y in train_loader:
            train_x = train_x.to(device)
            #print(f"train_x device: {train_x.device}")  # Debugging line
            train_y = torch.tensor(train_y, dtype=torch.float32).to(device).view(-1, 1)
            #print(f"train_y device: {train_y.device}")  # Debugging line
            pred = model(train_x)
            loss = criterion(pred, train_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            y_pred = np.round(pred.detach().cpu())
            train_correct += y_pred.eq(train_y.detach().cpu()).sum().item()

            #DEBUG
            if (progress_count % 10) == 0:
               print (y_pred.eq(train_y.detach().cpu()).sum().item(), len(y_pred))
            progress_count += 1

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = train_correct / len(train_loader.dataset)
        train_accuracies.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            for valid_x, valid_y in valid_loader:
                valid_x = valid_x.to(device)
                valid_y = valid_y.to(device).float()
                valid_y = valid_y.view(valid_y.size(0), -1)
                pred = model(valid_x)
                loss = criterion(pred, valid_y)
                valid_loss += loss.item()

                y_pred = np.round(pred.detach().cpu())
                valid_correct += y_pred.eq(valid_y.detach().cpu()).sum().item()

        valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(valid_loss)
        valid_accuracy = valid_correct / len(valid_loader.dataset)
        valid_accuracies.append(valid_accuracy)

        elapsed_time = time.time() - start
        print(
            f'[Epoch {epoch + 1}/{epochs}]: {elapsed_time:.3f} sec(elapsed time), train loss: {train_losses[-1]:.4f}, train acc: {train_accuracy * 100:.3f}% / valid loss: {valid_losses[-1]:.4f}, valid acc: {valid_accuracy * 100:.3f}%')

        if valid_losses[-1] < lowest_loss:
            lowest_loss = valid_losses[-1]
            lowest_epoch = epoch
            best_model = deepcopy(model.state_dict())
        else:
            if (early_stop > 0) and lowest_epoch + early_stop < epoch:
                print("Early Stopped", epoch, "epochs")
                break

        scheduler.step()

    model.load_state_dict(best_model)
    return model, lowest_loss, train_losses, valid_losses, train_accuracies, valid_accuracies

model, lowest_loss, train_losses, valid_losses, train_accuracies, valid_accuracies = train_model(model, loss_func, optimizer, 0, 1, train_batches, val_batches)

PATH = './'
torch.save(model.state_dict(), PATH + 'model_vit_base_patch32_224_in21k_linear_schedule_with_warmup_adam_1e5.pth')  # 모델 객체의 state_dict 저장

PATH = './'
model.load_state_dict(torch.load(PATH + 'model_vit_base_patch32_224_in21k_linear_schedule_with_warmup_adam_1e5.pth'))

# 테스트 데이터셋 로드
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
dataset_test = CustomDataset(test_list, transform=transforms_for_val_test, is_test=True)
test_batches = DataLoader(dataset=dataset_test, batch_size=64, shuffle=False)



def predict(model, data_loader):
    ids = list()
    with torch.no_grad():
        model.eval()
        ret = None
        for img, fileid in data_loader:
            img = img.to(device)
            pred = model(img)
            ids += list(fileid)
            if ret is None:
                ret = pred.cpu().numpy()
            else:
                ret = np.vstack([ret, pred.cpu().numpy()])
    return ret, ids




pred, ids = predict(model, test_batches)



print (pred.shape, len(ids))



# 캐글 제출 형식에 맞게 예측값 변환
submission = pd.DataFrame({'id': ids, 'label': np.clip(pred, 0.007, 1-0.007).squeeze()})
submission['id'] = submission['id'].str.split('.').str[0].astype(int)
submission.sort_values(by='id', inplace=True)
submission.reset_index(drop=True, inplace=True)
submission.to_csv('submission.csv', index=False)