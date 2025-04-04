from PIL import Image
import numpy as np

# function to crop the image into patches of size 100x100
def crop(im, default_size, crop_size):
    cropped_img = []
    cropped_img_size = []
    pos = []
    for j in range(0,default_size[1],crop_size[1]):
        for i in range(0, default_size[0], crop_size[0]):
            if default_size[1] - j < 100 or default_size[0] - i < 100:
                if default_size[1] - j < 100 and default_size[0] - i < 100:
                    box = (i,j,default_size[0],default_size[1])
                    pos.append(box)
                    a = im.crop(box)
                    a = a.resize(crop_size)
                    a = np.array(a)
                    cropped_img.append(a)
                    cropped_img_size.append((default_size[0] - i,default_size[1] - j))
                if default_size[0] - i < 100:
                    box = (i,j,default_size[0],j+crop_size[1])
                    pos.append(box)
                    a=im.crop(box)
                    a = a.resize(crop_size)
                    a = np.array(a)
                    cropped_img.append(a)
                    cropped_img_size.append((default_size[0] - i,crop_size[1]))
                else:
                    box = (i,j,i+crop_size[0],default_size[1])
                    pos.append(box)
                    a=im.crop(box)

                    a = a.resize(crop_size)
                    a = np.array(a)
                    cropped_img.append(a)
                    cropped_img_size.append((crop_size[0],default_size[1]-j))
            else:
                box = (i,j, i+crop_size[0],j+crop_size[1])
                pos.append(box)
                a = im.crop(box)
                a = np.array(a)
                cropped_img.append(a)
                cropped_img_size.append(crop_size)
    return np.array(cropped_img) / 127.5 - 1. , cropped_img_size, pos

def processImage(im, default_size):
    return crop(im, default_size=default_size, crop_size=(100,100))

# function to convert the patch to original size
def afterProcessing(t):
    t = 0.5 * t + 0.5
    k = t[0] * 255
    k = k.astype(np.uint8)
    k = Image.fromarray(k)
    return k

# function to save the patches to one image
def saveImgsToOne(o, default_size, pos):
    new_image = Image.new('RGB',(default_size[0] * 4, default_size[1] * 4), (250,250,250))
    for x in range(len(o)):
        new_image.paste(o[x],(pos[x][0] * 4,pos[x][1] * 4))
    return new_image