def net_structure(input):  # input: 100*100*1
    # 第一层卷积
    conv1_a = conv_layer(input, "conv1_a", 3, 3, 1, 128)  # 100*100*128,W[1]*128
    conv1_b = conv_layer(conv1_a, "conv1_b", 3, 3, 128, 128)  # 100*100*128,W[2]*128
    conv1_c = conv_layer(conv1_b, "conv1_c", 3, 3, 128, 128)  # 100*100*128,W[2]*128
    # 池化
    pool1 = tf.nn.max_pool(conv1_c, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")  # 50*50*128
    # dropout
    # drop1 = tf.nn.dropout(pool1, 0.5)  # 结构不变 大约总数的k倍的元素变成0，非零元素变成1/k倍

    # 第二层卷积
    conv2_a = conv_layer(pool1, "conv2_a", 3, 3, 128, 256)  # 50*50*256
    conv2_b = conv_layer(conv2_a, "conv2_b", 3, 3, 256, 256)  # 50*50*256
    conv2_c = conv_layer(conv2_b, "conv2_c", 3, 3, 256, 256)  # 50*50*256
    # 池化
    pool2 = tf.nn.max_pool(conv2_c, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")  # 25*25*256
    # dropout
    drop2 = tf.nn.dropout(pool2, 0.5)

    # 第三层卷积
    conv3_a = conv_layer(pool2, "conv3_a", 3, 3, 256, 512, padding="VALID")  # 6*6*512
    conv3_b = conv_layer(conv3_a, "conv3_b", 1, 1, 512, 256, padding="VALID")  # 6*6*256
    conv3_c = conv_layer(conv3_b, "conv3_c", 1, 1, 256, 128, padding="VALID")  # 6*6*128
    # 池化
    pool3 = tf.layers.average_pooling2d(conv3_c, 6, 1, name="pool3")  #

    # 全连接层
    fc = fc_layer(pool3, "fc", 7)

    return fc