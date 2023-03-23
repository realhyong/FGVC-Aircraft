import os
import shutil
from PIL import Image
import argparse
 
path = ''
 
image_path = path + '/images/'
save_train_path = path + '/dataset/train/'
save_test_path = path + '/dataset/test/'
save_trainval_path = path + '/dataset/trainval/'
save_val_path = path + '/dataset/val/'

imgs = os.listdir(image_path)
num = len(imgs)
 

f_test = open(path + ' ','r')
f_train = open(path + ' ','r')
f_trainval = open(path + ' ','r')
f_val = open(path + ' ','r')
test_list = list(f_test)
train_list = list(f_train)
trainval_list = list(f_trainval)
val_list = list(f_val)
 
parser = argparse.ArgumentParser(description='Data Split based on Txt')
parser.add_argument('--dataset',
                    default='test',
                    help='Select which dataset split, test, train, trainval, or val')
args = parser.parse_args()
 

print('==> data processing...')
if args.dataset == 'test':
    count = 0
    for i in range(num):
        aaaaa = len(test_list)
        bbbbbb = imgs[i][:7]
        for j in range(len(test_list)):
            if imgs[i][:7] == test_list[j][:7]:
                label = test_list[j][8:]
                label = label[:-1]
 
                if os.path.isdir(save_test_path + label):
                    shutil.copy(image_path + imgs[i], save_test_path + label + '/' + imgs[i])
                else:
                    os.makedirs(save_test_path + label)
                    shutil.copy(image_path + imgs[i], save_test_path+label+'/'+imgs[i])
                count += 1
                print('第%s张图片属于test类别' % count)
    print('Finished!!')
 
elif args.dataset == 'train':
    for i in range(num):
        for j in range(len(train_list)):
            if imgs[i][:7] == train_list[j][:7]:
                print('该图像属于train类别')
                # 获取类别标签
                label = train_list[j][8:]
                label = label[:-1]
 
                if os.path.isdir(save_train_path + label):
                    shutil.copy(image_path + imgs[i], save_train_path + label + '/' + imgs[i])
                else:
                    os.makedirs(save_train_path + label)
                    shutil.copy(image_path + imgs[i], save_train_path+label+'/'+imgs[i])
    print('Finished!!')
 
elif args.dataset == 'trainval':
    for i in range(num):
        for j in range(len(trainval_list)):
            if imgs[i][:7] == trainval_list[j][:7]:
                print('该图像属于trainval类别')
                # 获取类别标签
                label = trainval_list[j][8:]
                label = label[:-1]
 
                if os.path.isdir(save_trainval_path + label):
                    shutil.copy(image_path + imgs[i], save_trainval_path + label + '/' + imgs[i])
                else:
                    os.makedirs(save_trainval_path + label)
                    shutil.copy(image_path + imgs[i], save_trainval_path+label+'/'+imgs[i])
    print('Finished!!')
 
else:
    for i in range(num):
        for j in range(len(val_list)):
            if imgs[i][:7] == val_list[j][:7]:
                print('该图像属于val类别')
                # 获取类别标签
                label = val_list[j][8:]
                label = label[:-1]
 
                if os.path.isdir(save_val_path + label):
                    shutil.copy(image_path + imgs[i], save_val_path + label + '/' + imgs[i])
                else:
                    os.makedirs(save_val_path + label)
                    shutil.copy(image_path + imgs[i], save_val_path + label + '/' + imgs[i])
    print('Finished!!')