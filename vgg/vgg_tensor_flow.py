import tensorflow as tf

# VGG16
vgg16_model = tf.keras.applications.vgg16.VGG16(
    include_top=True, # include_top=True includes the fully connected layers for transfer tuple value to input_shape
    weights="imagenet", # will load pre-trained weights
    input_tensor=None,
    input_shape=None,
    poooling=None,
    classes=1000)

# VGG19
vgg19_model = tf.keras.applications.vgg19.VGG19(
    include_top=True,  # include_top=True includes the fully connected layers for transfer tuple value to input_shape
    weights="imagenet",  # will load pre-trained weights
    input_tensor=None,
    input_shape=None,
    poooling=None,
    classes=1000)