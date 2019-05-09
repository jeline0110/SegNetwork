import numpy
import os
import pickle as cPickle
from train import FLAGS
import cv2
import pdb
import numpy.random as nr

def mypickle(filename, data):
    fo = open(filename, "wb")
    cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()
    
def myunpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)

    fo = open(filename, 'rb')
    dict = cPickle.load(fo)    
    fo.close()
    return dict

#数据类，读取训练数据
class My_Data:
    def __init__(self):
    	self.img_size = 64
        self.img_dir = FLAGS.img_dir_train
        self.gt_dir = FLAGS.gt_dir_train
        self.sample_num = FLAGS.sample_num
        self.img_list, self.gt_list = self.data_list()
        self.idx = 0
        self.read_num = 1000 # 每次放到内存中的图片数量
        self.class_num = 2 # 分割类别 不算背景
        self.threshold = 0.5 # 阈值
        #if not os.path.exists(FLAGS.logs_dir): os.makedirs(FLAGS.logs_dir)
    
    #返回训练数据路径
    def data_list(self):
    	data_dic = './training_data_dic_%d' %self.sample_num
    	if not os.path.exists(data_dic):
	        img_list = []
	        gt_list = []
	        img_paths = np.sort(os.listdir(self.img_dir))
	        for img in img_paths[:self.sample_num]:
	            img_path = os.path.join(self.img_dir, img)
	            gt_path = os.path.join(self.gt_dir, img)
	            img_list.append(img_path)
	            gt_list.append(gt_path)
	        dic = {'img_list':img_list,'gt_list':gt_list}
	        mypickle(dic)
	     else:
	     	dic = myunpickle(data_dic)
	     	img_list = dic['img_list']
	     	gt_list = dic['gt_list']

        return img_list, gt_list
    
    # 将groundtruth转为训练可用的mask
    def gt_to_mask(self, gt):
    	height,width = gt.shape[:2]
    	mask = np.zeros((self.class_num,height,width), dtype=np.uint8)
    	for i in range(height):
    		for j in range(width):
    			if gt[i,j] != 0:
    				mask[i,j,gt[i,j]-1] = 1
    			else:
    				continue

    	return mask # [c,h,w]

    #加载数据
    def load_data(self, img_path, gt_path):
        img = cv2.imread(img_path)
        #gt = cv2.imread(gt_path,0) # 8位深度 单通道 0-255
        gt = cv2.imread(gt_path)
        if FLAGS.data_aug:
            if nr.rand()>0.1:
                img, gt = self.data_aug(img, gt)

        if FLAGS.debug:
            if os.path.exists(FLAGS.debug_dir):
                os.makedirs(FLAGS.debug_dir)
            img_name = os.path.dirname(img_path).split('/')[-1]
            gt_name = os.path.dirname(gt_path).split('/')[-1]
            i = len(os.listdir(FLAGS.debug_dir)) + 1
            gt_vis = gt/max(gt)*255.0 # 方便显示
            cv2.imwrite('./debug/%s_%d.png'%(img_name,i), img)
            cv2.imwrite('./debug/%s_%d.png'%(gt_name,i), gt_vis)
            #生成10张图片后暂停
            if i >= 10: 
                FLAGS.debug = False
                pdb.set_trace()

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA) 
        img = np.array(img, dtype=np.float32) - 127.5  

        gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)  
        mask = self.mask_to_net(gt)
        mask = np.array(mask, dtype=np.float32)

        return img, mask

    #以tensor形式读取数据
    def data_to_tensor(self):
        images = []
        maskes = []
        
        for i in range(self.read_num):
            if self.idx == self.sample_num:
                self.idx = 0
            img_path = self.img_list[self.idx]
            gt_path = self.gt_list[self.idx]
            img, mask = self.load_data(img_path,gt_path)
            img = img.transpose(2,0,1) # cv2:h,w,c -> torch:c,h,w

            images.append(img)
            maskes.append(mask)
            #if self.idx%1000 == 0: print('Data is loading!')
            self.idx += 1

        images = np.array(images).astype(np.float32)
        maskes = np.array(maskes).astype(np.float32)

        images = torch.from_numpy(images) #转为tensor
        maskes = torch.from_numpy(maskes)
        #print('Data loading completed!')

        return images, maskes

    # 数据增强函数
    def data_aug(self, img, gt):
        crop_pad = True
        rotate = True
        bright = True
        noise = True
        h,w,c = img.shape[:-1]

        if crop_pad:
            pad_w = max(int(0.1 * min(h,w)),1)
            img = np.pad(img, ((pad_w,pad_w),(pad_w,pad_w)),'constant',constant_vaules=(0))
            gt = np.pad(gt, ((pad_w,pad_w),(pad_w,pad_w)),'constant',constant_vaules=(0))
            start_x = nr.randint(0,2*pad_w)
            end_x = w + start_x
            start_y = nr.randint(0,2*pad_w)
            end_y = h + start_y
            img = img[start_y:end_y,start_x:end_x]
            gt = gt[start_y:end_y,start_x:end_x]

        if bright:
            bright_ratio = 0.80 + 0.40 * nr.rand()
            img = img * bright_ratio
            img[img>255] = 255
            img[img<0] = 0

        if noise:
            noise_mask = 10.0 - 20.0 * nr.rand(h,w,c)
            img = img + noise_mask
            img[img>255] = 255
            img[img<0] = 0

        if rotate:
            pad_w = max(int(0.3*min(h,w)))
            img = np.pad(img, ((pad_w,pad_w),(pad_w,pad_w)),'constant',constant_vaules=(0))
            gt = np.pad(gt, ((pad_w,pad_w),(pad_w,pad_w)),'constant',constant_vaules=(0))
            h_new,w_new = img.shape[:2]
            center = (h_new/2,w_new/2)
            angle = -2.0 + 4.0 * nr.rand()
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (h_new,w_new))
            gt = cv2.warpAffine(gt, M, (h_new,w_new))

            start_x = int(w_new/2 - w/2)
            end_x = w + start_x
            start_y = int(h_new/2 - h/2)
            end_y = h + start_y
            img = img[start_y:end_y,start_x:end_x]
            gt = gt[start_y:end_y,start_x:end_x]
        
        return img, gt 

#评估类 用于分割结果可视化
class val:
    def __init__(self):
        if not os.path.exists(FLAGS.val_out_dir): os.makedirs(FLAGS.val_out_dir)
    
    def load_data(img_path):
        img = cv2.imread(img_path)
        h,w = img.shape[:2]
        img = img.transpose(2,0,1)
        img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA) 
        img = np.array(img, dtype=np.float32) - 127.5
        img = img[np.newaxis,...]
        img = torch.from_numpy(img)

        return img,h,w

    # 将预测结果可视化
    def pred_to_vis(self,name,pred,h,w): # torch:[c,h,w]
        pred = pred.detach().cpu()
        pred = pred.numpy().squeeze(0)
        assert(pred.shape[0]==self.class_num) # 检查通道数
        channel,height,width = pred[:-1] # 通道idx按照类别idx排列
        vis = np.zeros((height,width), dtype=np.uint8)
        pred = np.reshape(pred,(channel,height*width))
        
        for i in range(height*width):
            temp = 0
            for j in range(channel):
                temp = max(temp,pred[j,i])  # 防止同一个像素点分类冲突

            label_idx = pred[:,i].idx(temp)+1 # 返回label值
            if temp >= self.threshold:
                vis[i/height,i%width] = label_idx
            else:
                continue

        vis = vis/max(vis)*255.0 # 方便显示
        vis = cv2.resize(vis, (h,w))
        cv2.imwrite(FLAGS.val_out_dir+'%s.png'%name,vis)
