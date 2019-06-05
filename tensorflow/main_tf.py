from data_processor_tf import *
from net_tf import *
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
parser.add_argument("--logs_dir", type=str, default="./training/", help="path to logs directory")
parser.add_argument("--mode", type=str, default="train", help="mode: train or test")
parser.add_argument("--fintune", type=bool, default=False, help="fintune model?")
parser.add_argument("--ckpt_path", type=str, default="./training/model.ckpt--1000", help="path to model ckpt.")
parser.add_argument("--img_dir_train", type=str, default="/home/scf/unet/images", help="path to imput images for training")
parser.add_argument("--gt_dir_train", type=str, default="/home/scf/unet/annotations", help="path to groundtruth for training")

parser.add_argument("--img_dir_test", type=str, default="/home/scf/unet/images", help="path to imput images for training")
#parser.add_argument("--gt_dir_test", type=str, default="/home/scf/unet/annotations", help="path to groundtruth for training")

parser.add_argument("--test_out_dir", type=str, default="./test/", help="path to output in the test phase")
parser.add_argument("--val_out_dir", type=str, default="./val/", help="path to output in the training process.")
parser.add_argument("--learning_rate", type=float, default="1e-4", help="Learning rate for Adam Optimizer")
parser.add_argument("--sample_num", type=int, default=300, help="How many samples to be used?")
parser.add_argument("--data_aug", type=bool, default=False, help="data augmentation.")
parser.add_argument("--debug", type=bool, default=False, help="debug flag.")
parser.add_argument("--debug_dir", type=str, default="./debug/", help="path to save debug image.")
parser.add_argument("--class_num", type=int, default=1, help="class number")
FLAGS = parser.parse_args()

MAX_STEP = 2000

def train():
    val = Val()
    my_data = My_Data()

    images = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="input_image")
    masks = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name="ground_truth")
    preds = seg_net(images)

    loss_func = tf.losses.mean_squared_error(labels=masks, predictions=preds)*64*64
    train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_func)
    
    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if not os.path.exists(FLAGS.logs_dir): os.makedirs(FLAGS.logs_dir)

    start_step = 0
    #选择是否进行fintune
    if FLAGS.fintune:
        saver.restore(sess, FLAGS.ckpt_path)
        print("Model restored...")
        start_step = int(FLAGS.ckpt_path.split('-')[1].split('.')[0])
        print("Continue train model at %d epoch" %(start_step))

    for step in range(start_step, MAX_STEP): # 分批次读取数据
        train_img, train_mask = my_data.data_to_tensor()
        feed_dict = {images: train_img, masks: train_mask}
        sess.run(train_op,feed_dict=feed_dict)
        pred = sess.run(preds,feed_dict)
        #pdb.set_trace()

        loss = sess.run(loss_func,feed_dict=feed_dict)

        if step % 50 == 0:
            print("Step:%d/%d, loss:%g" % ((step+1),MAX_STEP,loss))
    
            #输出中间结果，查看模型效果
            val_img,h,w = val.load_data('img.jpg')
            feed_dict = {images:val_img}
            pred = sess.run(preds,feed_dict)
            pred_name = str(step+1)
            val.pred_to_vis(pred_name, pred, h, w)

        if step % 1000 == 0 or loss < 5:
            print("Model saved")
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", step) #只保存模型参数

#测试函数
def test():
    if not os.path.exists(FLAGS.test_out_dir): os.makedirs(FLAGS.test_out_dir)
    val = Val()

    images = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input_image")
    preds = seg_net(images)

    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, FLAGS.ckpt_path)
    print("Model restored...")
    
    #测试一组图片
    i = 0
    for img in np.sort(os.listdir(FLAGS.img_dir_test)):
        img_path = os.path.join(FLAGS.img_dir_test, img)
        img_name = os.path.basename(img_path).split('.png')[0]
        test_img,h,w = val.load_data(img_path)
        feed_dict = {images:test_img}
        pred = sess.run(preds,feed_dict)
        val.pred_to_vis(img_name, pred, h ,w)
        i += 1
        print('the %d image has been test!'%i)

if __name__ == '__main__':
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'test':
        test()