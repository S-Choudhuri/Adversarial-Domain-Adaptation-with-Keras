import os
import argparse
import random
import numpy as np
from PIL import Image

def pil_loader(path):
    # Return the RGB variant of the input image
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
            
def data_loader(filepath, inp_dims):
    # Load images and corresponding labels from the text file and save them in a numpy array
    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit() 
    img = []
    label = []
    with open(filepath) as fp:
        for line in fp:
            token = line.split()
            i = pil_loader(token[0])
            i = i.resize((inp_dims[0],inp_dims[1]), Image.ANTIALIAS)
            img.append(i)
            label.append(int(token[1]))
    img = np.array(img)
    label = np.array(label)
    return img, label

def train(param):



if __name__ == "__main__":

	# Reading parameter values from the console
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="GPU id to run")
    parser.add_argument('--network_name', type=str, default='ResNet50', help="Name of the feature extractor network; ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--dataset_name', type=str, default='office', help="Name of the source dataset")
    parser.add_argument('--source_path', type=str, default='data/office/amazon_10_list.txt', help="Path to source dataset")
    parser.add_argument('--target_path', type=str, default='data/office/webcam_10_list.txt', help="Path to target dataset")
    parser.add_argument('--test_interval', type=int, default=500, help="Gap between two successive test phases")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="Gap between saving output models")
    parser.add_argument('--output_dir', type=str, default='models', help="Directory for saving output model")
    args = parser.parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Initialize parameters
    param = {}
    param["inp_dims"] = [224, 224, 3]
    param["num_iterations"] = 12004
    param["test_interval"] = args.test_interval
    param["source_path"] = args.source_path
    param["target_path"] = args.target_path
    param["snapshot_interval"] = args.snapshot_interval
    param["output_for_test"] = True
    param["output_path"] = "snapshot/" + args.output_dir

    # Create directory for saving models and log files
    if not os.path.exists(param["output_path"]):
        os.mkdir(param["output_path"])
    param["output_file"] = open(os.path.join(param["output_path"], "log.txt"), "w")
    if not os.path.exists(param["output_path"]):
        os.mkdir(param["output_path"])

    # Load source and larget data
    param["s_data"], param["s_label"] = data_loader(param["source_path"], param["inp_dims"])
    param["t_data"], param["t_label"] = data_loader(param["target_path"], param["inp_dims"])

    #train data
    train(param)


