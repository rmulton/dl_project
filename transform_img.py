"""
This script is used to shrink the dataset size.

Use it with the following command :
$ python transform_img.py -from_dir <source folder> -to_dir <destination folder>

"""

import os
from imageio import imread, imsave
from skimage.transform import resize
import argparse

if __name__=="__main__":
    # Configuration with command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-from_dir", type=str, help="folder to resize images from")
    parser.add_argument("-to_dir", type=str, help="folder to resize images to")
    args = parser.parse_args()

    # Check that the arguments are not empty
    if not args.from_dir or not args.to_dir:
        print("One of the arguments is empty")
        raise ValueError

    # Get the arguments
    CURRENT_FOLDER = args.from_dir
    NEW_FOLDER = args.to_dir

    # Create the new_folder if necessary
    if not NEW_FOLDER in os.listdir():
        print("Creating the folder {}".format(NEW_FOLDER))
        os.mkdir(NEW_FOLDER)

    # Resize the images
    for filepath in os.listdir(CURRENT_FOLDER):

        # Print the current file
        print("Working on {}".format(filepath))

        # Warning for non JPEG
        if not (filepath[-4:]=='.jpg' or filepath[-5:]=='.jpeg'):
            print("Warning: \"{}\" does not seem to be a JPEG file".format(filepath))

        # Read the source image
        img_path = os.path.join(CURRENT_FOLDER, filepath)
        img = imread(img_path)

        # Resize the image
        resized = resize(img, (256, 256))
        new_path = os.path.join(NEW_FOLDER, filepath)

        # Save the resized image to the destination
        imsave(new_path, resized)

    print("Done")

