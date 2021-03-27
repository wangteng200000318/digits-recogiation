import tensorflow as tf
import numpy as np

from  PIL import Image
import matplotlib.pylab as plt
batch_size = 128
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batch_size * 5).batch(batch_size)

test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batch_size)


class AE(tf.keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(20)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(784)
        ])

    def call(self, inputs, training=None, mask=None):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)
        return x_hat

plt.imshow(x_test[0])
plt.show()
model = AE()
model.build(input_shape=(4, 784))
optimizer = tf.optimizers.Adam()
for epoch in range(100):
    for step, x in enumerate(train_db):
        x = tf.reshape(x, [-1, 784])

        with tf.GradientTape() as tape:
            x_rec_logits = model(x)
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_mean(rec_loss)
        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step % 100 == 0:
            print(epoch, step, float(rec_loss))

x = next(iter(test_db))
logits = model(tf.reshape(x, [-1, 784]))  # 打平并送入自编码器
x_hat = tf.sigmoid(logits)  # 将输出转换为像素值，使用sigmoid函数 # 恢复为 28x28,[b, 784] => [b, 28, 28]
x_hat = tf.reshape(x_hat, [-1, 28, 28])
# 输入的前50张+重建的前50张图片合并，[b, 28, 28] => [2b, 28, 28]
x_concat = tf.concat([x[:50], x_hat[:50]], axis=0)
x_concat = x_concat.numpy() * 255.  # 恢复为0~255范围
x_concat = x_concat.astype(np.uint8)  # 转换为整型
def save_images(imgs, name):
# 创建280x280大小图片阵列
    new_im = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28): # 10 行图片阵列
        for j in range(0, 280, 28): # 10 列图片阵列
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j)) # 写入对应位置
            index += 1 # 保存图片阵列
            new_im.save(name)
save_images(x_concat, 'ae_images/rec_epoch_%d.png' % epoch)  # 保存图片
