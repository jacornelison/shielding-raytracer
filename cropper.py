from PIL import Image
import os.path, sys

path = "C:/Users/James/Documents/GitHub/postings/2020mmdd_mirror_check2/figs"
dirs = os.listdir(path)



def crop():
    for item in dirs:
        fullpath = os.path.join(path,item)         #corrected
        if item[0:3]=="pos":

        #if False:#os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop()
            imCrop = im.crop((100, 0, 1200, 890)) #corrected
            imCrop.save(f + '_cropped.png', quality=100)

crop()