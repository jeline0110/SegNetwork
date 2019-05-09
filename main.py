# coding=utf-8
# author: LianJie
# created: 2019.5.6
# version: 1
# description: Using the architecture of unet to build an image segmentation network.
# environment: pytorch:1.0 python:3.5

from data_processor import *
from net import *
import argparse
from torch.autograd import Variable
import torch.utils.data as Data 
import pdb
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=4, help="batch size for training")
parser.add_argument("--logs_dir", type=str, default="./training/", help="path to logs directory")
parser.add_argument("--mode", type=str, default="train", help="mode: train or test")
parser.add_argument("--fintune", type=bool, default=False, help="fintune model?")
parser.add_argument("--ckpt_path", type=str, default="./training/model-10.pth", help="path to model ckpt.")
parser.add_argument("--img_dir_train", type=str, default="/home/scf/gan/img/train/img", help="path to imput images for training")
parser.add_argument("--gt_dir_train", type=str, default="/home/scf/gan/img/train/gt/", help="path to groundtruth for training")
'''
parser.add_argument("img_dir_test", "/home/scf/gan/img/test/MRI/", "path to input images for test")
parser.add_argument("gt_dir_test", "/home/scf/gan/img/test/ct_label/", "path to groundturh for test")
'''
parser.add_argument("--test_out_dir", type=str, default="./test/", help="path to output in the test phase")
parser.add_argument("--val_out_dir", type=str, default="./val/", help="path to output in the training process.")
parser.add_argument("--learning_rate", type=float, default="1e-4", help="Learning rate for Adam Optimizer")
parser.add_argument("--sample_num", type=int, default=1, help="How many samples to be used?")
parser.add_argument("--data_aug", type=bool, default=True, help="data augmentation.")
parser.add_argument("--debug", type=bool, default=True, help="debug flag.")
parser.add_argument("--debug_dir", type=str, default="./debug/", help="path to save debug image.")
FLAGS =  parser.parse_args()

MAX_EPOCH = 20 

#适用于样本极度不均的情况
class DiceLoss(nn.Module):
    def __init__(self, class_num=2, smooth=0.0001):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(self.class_num):
            input_i = input[:, i, :, :]
            target_i = target[:, i, :, :]
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += 1 - dice
        dice_loss = Dice / (self.class_num)

        return dice_loss

def train():
    val = Val()
    my_data = My_Data()
    my_unet = My_Unet().cuda()
    loss_func = torch.nn.BCEloss().cuda()
    optimizer = torch.optim.Adam(my_unet.parameters(), lr=FLAGS.learning_rate, betas=(0.5, 0.999))
    
    start_epoch = 0
    #选择是否进行fintune
    if FLAGS.fintune:
        my_unet.load_state_dict(torch.load(FLAGS.ckpt_path))
        print("Model restored...")
        start_epoch = int(FLAGS.ckpt_path.split('-')[1].split('.')[0])
        print("Continue train model at %d epoch" %(start_epoch))

    for epoch in range(start_epoch, MAX_EPOCH):
        count = int(FLAGS.sample_num/my_data.read_num) + 1
        for i in range(count): # 分批次读取数据
            train_img, train_mask = my_data.data_to_tensor() # 数据转为tensor类型
            torch_dataset = Data.TensorDataset(train_img, train_mask) # 将数据转为torch可以识别的数据类型
            loader = Data.DataLoader(dataset=torch_dataset, batch_size=FLAGS.batch_size, shuffle=True) # 将数据放入loader中
    
            for step, (batch_x, batch_y) in enumerate(loader):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                score_map = my_unet(batch_x)
                loss = loss_func(score_map.view(-1),batch_y.view(-1))
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            
                if step % 10 == 0:
                    print("Epoch:%d/%d, Number:%d/%d, Batch:%d/%d, loss:%g" % 
                        ((epoch+1),MAX_EPOCH,(i+1),count,(step+1),len(loader),loss))
        
            #输出中间结果，查看模型效果
            val_img,h,w = val.load_data('val.png').cuda()
            pred = my_unet(val_img)
            pred_name = str(epoch+1)+'-'+str(step)
            val.pred_to_vis(pred_name, pred, h, w)
            
        print("Model saved")
        torch.save(my_unet.state_dict(), FLAGS.logs_dir+"model-%d.pth"%(epoch+1)) #只保存模型参数

#测试函数
def test():
    if not os.path.exists(FLAGS.test_out_dir): os.makedirs(FLAGS.test_out_dir)
    val = Val()
    my_unet = My_Unet().cuda()
    my_unet.load_state_dict(torch.load(FLAGS.ckpt_path))
    print("Model restored...")
    
    #测试一组图片
    i = 0
    for img in np.sort(os.listdir(FLAGS.img_dir_test)):
        img_path = os.path.join(FLAGS.img_dir_test, img)
        img_name = os.path.basename(img_path).split('.png')[0]
        test_img,h,w = val.load_data(img_path).cuda()
        pred = my_unet(test_img)
        val.pred_to_vis(img_name, pred, h ,w)
        i += 1
        print('the %d image has been test!'%i)

if __name__ == '__main__':
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'test':
        test()

