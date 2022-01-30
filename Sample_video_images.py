import random
from PIL import Image

def main():
    #923
    rand_list1 = random.sample(range(0, 923), 100)
    #1379
    rand_list2 = random.sample(range(0, 1378), 100)

    path = "/Users/zhangyuke/PycharmProject/Merck_CV/"
    i = 1
    for e1 in rand_list1:
        # Importing Image module from PIL package
        # creating a image object (main image)
        im1 = Image.open(f"{path}data{i}/frame{e1}.jpg")
        # save a image using extension
        im1 = im1.save(f"{path}selected_data{i}/v{i}_frame{e1}.jpg")
    i = 2
    for e2 in rand_list2:
        # Importing Image module from PIL package
        # creating a image object (main image)
        im2 = Image.open(f"{path}data{i}/frame{e2}.jpg")
        # save a image using extension
        im2 = im2.save(f"{path}selected_data{i}/v{i}_frame{e2}.jpg")

main()