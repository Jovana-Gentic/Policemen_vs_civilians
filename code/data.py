from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def create_resolution():
    resolution = (150, 150)
    return resolution

def create_dataset():
    # Since we have only a few samples, we use data augementation because of overfitting
    # We currently use random rotations, horizontal flips and random brightness modifications
    # TODO try different augmentation paramaters
    train_data = ImageDataGenerator(rescale=1/255,
                                    rotation_range=15,
                                    fill_mode='constant',
                                    cval=0,
                                    horizontal_flip=True,
                                    brightness_range=[0.4, 1])
    val_data = ImageDataGenerator(rescale=1/255)

    # As a form of regularisation we use small batch size so we get more training steps per epoch
    batch = 8
    resolution = create_resolution()
    train_data = train_data.flow_from_directory(r"../data/train",
                                                target_size=resolution,
                                                class_mode='binary',
                                                batch_size=batch)
    val_data = val_data.flow_from_directory(r"../data/val",
                                            target_size=resolution,
                                            class_mode='binary',
                                            batch_size=batch)
    return train_data, val_data, batch

def get_positives_and_negatives(val_data):
    # We separate inputs inputs in positives and negatives so we can plot it

    inputs = []
    labels = []
    for i, test_batch in enumerate(val_data):
        inputs.append(test_batch[0])
        labels.append(test_batch[1])
        if i >= len (val_data):
            break


    inputs = np.concatenate(inputs, axis=0)
    labels = np.concatenate(labels, axis=0)
    pos_inputs = inputs[labels == 1.]
    neg_inputs = inputs[labels == 0.] 
    return pos_inputs, neg_inputs, inputs, labels