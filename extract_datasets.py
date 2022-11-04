import os
import re
import random
from shutil import copyfile, move
from PIL import Image


data_path = "Final_project/data"
landscapes_path = "Final_project/data/landscapes"
dst_path = "Final_project/data/landscapes_labels"


def extract_from_label(label):
    files = os.listdir(landscapes_path)
    files.remove("others")

    if label != 1:
        label = "(" + str(label) + ")"
        files = [f for f in files if label in f]

    else:
        files = [f for f in files if "(" not in f]

    return files


def extract_all_labels():

    others_path = landscapes_path + "/" + "others"

    if "others" not in os.listdir(landscapes_path):
        os.makedirs(others_path)

    for label in range(1, 8, 1):
        print("Label ", label)
        files = extract_from_label(label)
        print(len(files))

        dst_folder_path = dst_path + "/" + str(label)

        if dst_folder_path not in os.listdir(data_path):
            os.makedirs(dst_folder_path)

        for f in files:
            src = landscapes_path + "/" + f
            with Image.open(src) as im:
                width, height = im.size

            if width < 384 or height < 384:
                dst = others_path + "/" + f
                move(src, dst)

            else:
                dst = dst_folder_path + "/" + f
                copyfile(src, dst)


def extract_sample():
    files = os.listdir(landscapes_path)
    files.remove("others")
    sample = random.sample(files, 550)

    dst_folder_path = dst_path + "/" + "sample"

    if dst_folder_path not in os.listdir(data_path):
        os.makedirs(dst_folder_path)

    for f in sample:
        src = landscapes_path + "/" + f
        dst = dst_folder_path + "/" + f
        copyfile(src, dst)


if __name__ == "__main__":
    # Separate the landscapes dataset into classes
    extract_all_labels()

    print("Extracting sample...")
    extract_sample()
