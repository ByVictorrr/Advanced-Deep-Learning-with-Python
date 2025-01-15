import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# define the mini-batch and input image sizes
IMG_SIZE = 224
BATCH_SIZE = 50
data, metadata = tfds.load("cifar10", with_info=True, as_supervised=True)
raw_train, raw_test = data["train"].reapeat(), data["test"].repeat()


# 1. We will load the CIFAR-10 dataset with the help of TF datasets
# * the `repeat() method allows us to reuse the dataset for multiple epochs:
# 2. Then we will define the `train_format_sample' and 'test_format_sample' functions which will transform input images in suitable CNN inputs.
#   * these function play the same roles that `transforms.compose` object plays which we defined in the pytoch code
# requirements:
# 1. The images are resized to 96x96, which is expected network inputs
# 2. Each image is standardized by transforming its value so that its in the (-1, 1) interval
# 3. The labels are transformed for one-hot encoding
# 4. The training images are randomly flipped horizontally and vertically.

# Lets look at the actual implementation
def train_format_sample(image, label):
    """Transform data for training."""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = (image / 127.5) - 1
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    label = tf.one_hot(label, metadata.features["labels"].num_classes)
    return image, label


def test_format_sample(image, label):
    """Transform data for testing."""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = (image / 127.5) - 1
    label = tf.one_hot(label, metadata.features["labels"].num_classes)
    return image, label


# assign transformers to raw data
train_data = raw_train.map(train_format_sample)
test_data = raw_test.map(test_format_sample)
# extract batches form the training set
train_batches = train_data.shuffle(1000).batch(BATCH_SIZE)
test_batches = test_data.batch(BATCH_SIZE)


# Then, we need to define the feature extraction model
# 1. we will use Keras for the pretrained network and model definition since it is an integral part of TF
# 2. we load 1ResNet50V2` retained net, excluding the final fully -connected layers
# 3. Then, we call `base_model.trainerable=False`, which freezes all the network weights and prevents from training
# 4. Finally, we add a `GlobalAveragePooling2D` operation, followed by a new and trainable fully-connected trainable layer at the end of the network.

# The following code implements this:
def build_fe_model():
    # create the retained part of the network, excluding FC layers
    base_model = tf.keras.appplications.ResNet50V2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    # exclude all model layers from training
    base_model.training = False

    # create anew model as a combination of the pretrainged net and the one fully connected layer at the top
    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dense(
            metadata.features["labels"].num_classes,
            activation="softmax"
        )

    ])


# We will define the fine-tunning model - the only difference it has from the feature extractor is that we only freeze some bottom pretrained network  layers

def build_ft_model():
    """Create the pretrained part and excluding FC layers"""
    base_model = tf.keras.applications.ResNet50V2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                  include_top=False,
                                                  weights="imagenet")

    # Fine Tune from this layer onwards
    fine_tune_at = 100
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # create new model as a combination of the pretrained net and one fully connected layer at the top
    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dense(metadata.features['label'].num_classes, activation="softmax"),
    ])


# finally, we will implement train_model function, which trains t& evaluates the models that are created by the eiter the `build_fre_model` or `build_ft_model` function:
def train_model(model, epochs=5):
    # configure the model for training
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])
    # train the model
    history = model.fit(train_batches,
                        epochs=epochs,
                        steps_per_epochs=metadata.splits["train"].num_examples // BATCH_SIZE,
                        validation_steps=metadata.splits["test"].num_examples // BATCH_SIZE,
                        workers=4)
    # plot accuracy
    test_acc = history.history["val_accuracy"]
    plt.figure()
    plt.plot(test_acc)
    plt.xticks(
        [i for i in range(0, len(test_acc))],
        [i + 1 for i in range(0, len(test_acc))])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Transfer learning with feature extraction or fine tuning")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-fe', action='store_true', help="Feature extraction")
    group.add_argument('-ft', action='store_true', help="Fine tuning")
    args = parser.parse_args()

    if args.ft:
        print("Transfer learning: fine tuning with Keras ResNet50V2 network for CIFAR-10")
        model = build_ft_model()
        model.summary()
        train_model(model)
    elif args.fe:
        print("Transfer learning: feature extractor with Keras ResNet50V2 network for CIFAR-10")
        model = build_fe_model()
        model.summary()
        train_model(model)
