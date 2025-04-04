# import libraries
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

# importing custom functions
from utils import processImage, afterProcessing, saveImgsToOne

# class for SRGAN (Super-Resolution Generative Adversarial Network)
class SRGAN:
    def __init__(self, model_path = 'weights/SRGAN_EP50000.h5'):
        self.model_path = model_path
        self.__load_model()
        pass

    def __load_model(self):
        # check if the model file exists
        if os.path.exists(self.model_path) == False:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # load the model
        print("Loading model from path: ", self.model_path)
        self.model = load_model(self.model_path)
        print("Model loaded successfully.")

    def predict(self, image):
        # get patches of the image
        im, _, pos = processImage(image, image.size)
        
        # upscale the patches using the model
        processedImgList = []
        for x in range(len(im)):
            temp = tf.expand_dims(im[x], axis=0)
            result = self.model.predict(temp)
            out_patch = afterProcessing(result)
            processedImgList.append(out_patch)
        
        return saveImgsToOne(processedImgList, image.size , pos)


if __name__ == "__main__":
    # Example usage of the SRGAN class
    from PIL import Image

    # Load an image
    img_path = "input\lr_image.png"  # replace with your image path
    image = Image.open(img_path).convert("RGB")

    # Create an instance of the SRGAN class
    srgan = SRGAN()

    # Predict the super-resolved image
    super_resolved_image = srgan.predict(image)

    # Save or display the super-resolved image
    super_resolved_image.save("output\sr_image.png")