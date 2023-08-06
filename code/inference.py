import numpy as np
import cv2
from keras.preprocessing import image
from matplotlib import pyplot as plt
import tensorflow as tf
from data import create_resolution

# Loading model
checkpoint_filepath = '../model_checkpoint/model11'
model = tf.keras.models.load_model(checkpoint_filepath)

def test(filename, resolution, threshold):
    # Run inference on one image

    # Reading the image
    test_picture = image.load_img(f"../testpics/{filename}",target_size = resolution)
    x1=image.img_to_array(test_picture)
    x1=np.expand_dims(x1, axis=0)
    images = np.vstack([x1])/255

    # Making prediction
    ans = model.predict(images)
    if ans[0][0] <= threshold:
        policeman = "This is not a policeman!"
    else:
        policeman = "This is a policeman!"
    print(ans)

    # Display image
    img = cv2.imread(f"../testpics/{filename}")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{policeman} \n Probability: {ans[0][0]:.4f}")
    plt.show()

threshold = float(input("Enter model threshold: "))
resolution = create_resolution()
while True:
    test(input("Enter picture name: "), resolution, threshold)
