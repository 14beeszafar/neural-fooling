import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3, vgg19, vgg16
from keras import backend as K
from PIL import Image

model = vgg19.VGG19()

img = image.load_img("ca2.jpg", target_size=(224, 224))
input_image = image.img_to_array(img)

input_image = np.expand_dims(input_image, axis=0)

predictions = model.predict(input_image)

predicted_classes = vgg19.decode_predictions(predictions, top=1)
imagenet_id, name, confidence = predicted_classes[0][0]
print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))

input_image /= 255.
input_image -= 0.5
input_image *= 2.

model_input_layer = model.layers[0].input
model_output_layer = model.layers[-1].output

object_type_to_fake = 859
original_image = input_image;

max_change_above = original_image + 0.09
max_change_below = original_image - 0.09

hacked_image = np.copy(original_image)

learning_rate = 10

cost_function = model_output_layer[0, object_type_to_fake]

gradient_function = K.gradients(cost_function, model_input_layer)[0]

grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

cost = 0.0

while cost < 0.6:
    
    cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])
    learning_rate = 100
    
    hacked_image += gradients * learning_rate

    hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
    hacked_image = np.clip(hacked_image, -1.0, 1.0)

    print("Model's predicted likelihood that the image is a toaster: {:.8}%".format(cost * 100))

img = hacked_image[0]
img /= 2.
img += 0.5
img *= 255.

im = Image.fromarray(img.astype(np.uint8))
im.save("hacked-image.png")