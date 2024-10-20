import tensorflow as tf

print(tf.__version__)
print(tf.test.gpu_device_name())
print(tf.config.set_visible_devices)
print(tf.test.is_gpu_available())