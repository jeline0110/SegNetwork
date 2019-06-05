import tensorflow as tf
slim = tf.contrib.slim

def seg_net(image):
    net0 = slim.conv2d(image, 128, [5, 5], padding='SAME', scope='conv0')
    net = slim.conv2d(net0, 128, [3, 3], stride=2, padding='SAME', scope='conv1')
    net1 = slim.conv2d(net, 128, [3, 3], padding='SAME', scope='conv2')
    #32x32
    #pdb.set_trace()
    net = slim.conv2d(net1, 128, [3, 3], stride=2, padding='SAME', scope='conv3')
    net2 = slim.conv2d(net, 128, [3, 3], padding='SAME', scope='conv4')
    #16x16
    
    net = slim.conv2d(net2, 128, [3, 3], stride=2, padding='SAME', scope='conv5')
    net3 = slim.conv2d(net, 128, [3, 3], padding='SAME', scope='conv6')
    #8x8
    
    net = slim.conv2d(net3, 128, [3, 3], stride=2, padding='SAME', scope='conv7')
    net4 = slim.conv2d(net, 128, [3, 3], padding='SAME', scope='conv8')
    #4x4
    
    net5 = slim.conv2d(net4, 128, [4, 4], stride=2, padding='VALID', scope='conv9')
    #1x1
    
    net = slim.conv2d_transpose(net5, 128, [4, 4], stride=2, padding='VALID', scope='deconv10')
    net6 = slim.conv2d(net, 128, [3, 3], padding='SAME', scope='conv11')
    #4x4
    
    con6 = tf.concat([net6, net4],axis=3)
    net = slim.conv2d_transpose(con6, 128, [3, 3], stride=2, padding='SAME', scope='deconv12')
    net7 = slim.conv2d(net, 128, [3, 3], padding='SAME', scope='conv13')
    #8x8
    
    con7 = tf.concat([net7, net3],axis=3)
    net = slim.conv2d_transpose(con7, 128, [3, 3], stride=2, padding='SAME', scope='deconv14')
    net8 = slim.conv2d(net, 128, [3, 3], padding='SAME', scope='conv15')
    #16x16
    
    con8 = tf.concat([net8, net2],axis=3)
    net = slim.conv2d_transpose(con8, 128, [3, 3], stride=2, padding='SAME', scope='deconv16')
    net9 = slim.conv2d(net, 128, [3, 3], padding='SAME', scope='conv17')
    #32x32
    
    con9 = tf.concat([net9, net1],axis=3)
    net = slim.conv2d_transpose(con9, 128, [3, 3], stride=2, padding='SAME', scope='deconv18')
    net10 = slim.conv2d(net, 128, [3, 3], padding='SAME', scope='conv19')
    #64x64

    score_map = slim.conv2d(net10, 1, [1, 1], padding='VALID', activation_fn=tf.nn.sigmoid, scope='conv20')
    #pdb.set_trace()

    return score_map