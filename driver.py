SEED = 7
import os
import sys
import argparse
import random
import numpy as np
from tensorflow import set_random_seed

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
set_random_seed(SEED)
random.seed(SEED)

from PIL import Image
from keras.utils import to_categorical
from keras.layers import Input
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from sklearn.metrics import accuracy_score
import model
import optimizer

def pil_loader(path):
    # Return the RGB variant of input image
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def one_hot_encoding(param):
    # Read the source and target labels from param
    s_label = param["source_label"]
    t_label = param["target_label"]

    # Encode the labels into one-hot format
    classes = (np.concatenate((s_label, t_label), axis = 0))
    num_classes = np.max(classes)
    if 0 in classes:
            num_classes = num_classes+1
    s_label = to_categorical(s_label, num_classes = num_classes)
    t_label = to_categorical(t_label, num_classes = num_classes)
    return s_label, t_label
            
def data_loader(filepath, inp_dims):
    # Load images and corresponding labels from the text file, stack them in numpy arrays and return
    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit() 
    img = []
    label = []
    with open(filepath) as fp:
        for line in fp:
            token = line.split()
            i = pil_loader(token[0])
            i = i.resize((inp_dims[0], inp_dims[1]), Image.ANTIALIAS)
            img.append(np.array(i))
            label.append(int(token[1]))
    img = np.array(img)
    label = np.array(label)
    return img, label

def batch_generator(data, batch_size):
    #Generate batches of data.
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size = batch_size, replace = False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr

def train(param):
    models = {}
    inp = Input(shape = (param["inp_dims"]))
    embedding = model.build_embedding(param, inp)
    classifier = model.build_classifier(param, embedding)
    discriminator = model.build_discriminator(param, embedding)

    if param["number_of_gpus"] > 1:
        models["combined_classifier"] = multi_gpu_model(model.build_combined_classifier(inp, classifier), gpus = param["number_of_gpus"])
        models["combined_discriminator"] = multi_gpu_model(model.build_combined_discriminator(inp, discriminator), gpus = param["number_of_gpus"])
        models["combined_model"] = multi_gpu_model(model.build_combined_model(inp, [classifier, discriminator]), gpus = param["number_of_gpus"])
    else:
        models["combined_classifier"] = model.build_combined_classifier(inp, classifier)
        models["combined_discriminator"] = model.build_combined_discriminator(inp, discriminator)
        models["combined_model"] = model.build_combined_model(inp, [classifier, discriminator])

    models["combined_classifier"].compile(optimizer = optimizer.opt_classifier(param), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    models["combined_discriminator"].compile(optimizer = optimizer.opt_discriminator(param), loss = 'binary_crossentropy', metrics = ['accuracy'])
    models["combined_model"].compile(optimizer = optimizer.opt_combined(param), loss = {'class_act_last': 'categorical_crossentropy', 'dis_act_last': \
        'binary_crossentropy'}, loss_weights = {'class_act_last': param["class_loss_weight"], 'dis_act_last': param["dis_loss_weight"]}, metrics = ['accuracy'])

    Xs, ys = param["source_data"], param["source_label"]
    Xt, yt = param["target_data"], param["target_label"]

    # Source domain is represented by label 0 and Target by 1
    ys_adv = np.array(([0.] * ys.shape[0]))
    yt_adv = np.array(([1.] * yt.shape[0]))

    y_advb_1 = np.array(([1] * param["batch_size"] + [0] * param["batch_size"])) # For gradient reversal
    y_advb_2 = np.array(([0] * param["batch_size"] + [1] * param["batch_size"]))
    weight_class = np.array(([1] * param["batch_size"] + [0] * param["batch_size"]))
    weight_adv = np.ones((param["batch_size"] * 2,))
    S_batches = batch_generator([Xs, ys], param["batch_size"])
    T_batches = batch_generator([Xt, np.zeros(shape = (len(Xt),))], param["batch_size"])

    param["target_accuracy"] = 0

    optim = {}
    optim["iter"] = 0
    optim["acc"] = ""
    optim["labels"] = np.array(Xt.shape[0],)
    gap_last_snap = 0

    for i in range(param["num_iterations"]):        
        Xsb, ysb = next(S_batches)
        Xtb, ytb = next(T_batches)
        X_adv = np.concatenate([Xsb, Xtb])
        y_class = np.concatenate([ysb, np.zeros_like(ysb)])

        adv_weights = []
        for layer in models["combined_model"].layers:
            if (layer.name.startswith("dis_")):
                adv_weights.append(layer.get_weights())
          
        stats1 = models["combined_model"].train_on_batch(X_adv, [y_class, y_advb_1],\
                                sample_weight=[weight_class, weight_adv])            
        k = 0
        for layer in models["combined_model"].layers:
            if (layer.name.startswith("dis_")):                    
                layer.set_weights(adv_weights[k])
                k += 1

        class_weights = []        
        for layer in models["combined_model"].layers:
            if (not layer.name.startswith("dis_")):
                class_weights.append(layer.get_weights())  

        stats2 = models["combined_discriminator"].train_on_batch(X_adv, [y_advb_2])

        k = 0
        for layer in models["combined_model"].layers:
            if (not layer.name.startswith("dis_")):
                layer.set_weights(class_weights[k])
                k += 1

        if ((i + 1) % param["test_interval"] == 0):
            ys_pred = models["combined_classifier"].predict(Xs)
            yt_pred = models["combined_classifier"].predict(Xt)
            ys_adv_pred = models["combined_discriminator"].predict(Xs)
            yt_adv_pred = models["combined_discriminator"].predict(Xt)

            source_accuracy = accuracy_score(ys.argmax(1), ys_pred.argmax(1))              
            target_accuracy = accuracy_score(yt.argmax(1), yt_pred.argmax(1))
            source_domain_accuracy = accuracy_score(ys_adv, np.round(ys_adv_pred))              
            target_domain_accuracy = accuracy_score(yt_adv, np.round(yt_adv_pred))

            log_str = "iter: {:05d}: \nLABEL CLASSIFICATION: source_accuracy: {:.5f}, target_accuracy: {:.5f}\
                    \nDOMAIN DISCRIMINATION: source_domain_accuracy: {:.5f}, target_domain_accuracy: {:.5f} \n"\
                                                         .format(i, source_accuracy*100, target_accuracy*100,
                                                      source_domain_accuracy*100, target_domain_accuracy*100)
            print(log_str)

            if param["target_accuracy"] < target_accuracy:              
                optim["iter"] = i
                optim["acc"] = log_str
                optim["labels"] = ys_pred.argmax(1)

                if (gap_last_snap >= param["snapshot_interval"]):
                    gap_last_snap = 0
                    np.save(os.path.join(param["output_path"],"yPred_{}".format(optim["iter"])), optim["labels"])
                    open(os.path.join(param["output_path"], "acc_{}.txt".format(optim["iter"])), "w").write(optim["acc"])
                    models["combined_classifier"].save(os.path.join(param["output_path"],"iter_{:05d}_model.h5".format(i)))
        gap_last_snap = gap_last_snap + 1;

if __name__ == "__main__":
    # Read parameter values from the console
    parser = argparse.ArgumentParser(description = 'Domain Adaptation')
    parser.add_argument('--number_of_gpus', type = int, nargs = '?', default = '1', help = "Number of gpus to run")
    parser.add_argument('--network_name', type = str, default = 'ResNet50', help = "Name of the feature extractor network")
    parser.add_argument('--dataset_name', type = str, default = 'Office', help = "Name of the source dataset")
    parser.add_argument('--dropout_classifier', type = float, default = 0.25, help = "Dropout ratio for classifier")
    parser.add_argument('--dropout_discriminator', type = float, default = 0.25, help = "Dropout ratio for discriminator")    
    parser.add_argument('--source_path', type = str, default = 'amazon_10_list.txt', help = "Path to source dataset")
    parser.add_argument('--target_path', type = str, default = 'webcam_10_list.txt', help = "Path to target dataset")
    parser.add_argument('--lr_classifier', type = float, default = 0.0001, help = "Learning rate for classifier model")
    parser.add_argument('--b1_classifier', type = float, default = 0.9, help = "Exponential decay rate of first moment \
                                                                                             for classifier model optimizer")
    parser.add_argument('--b2_classifier', type = float, default = 0.999, help = "Exponential decay rate of second moment \
                                                                                            for classifier model optimizer")
    parser.add_argument('--lr_discriminator', type = float, default = 0.00001, help = "Learning rate for discriminator model")
    parser.add_argument('--b1_discriminator', type = float, default = 0.9, help = "Exponential decay rate of first moment \
                                                                                             for discriminator model optimizer")
    parser.add_argument('--b2_discriminator', type = float, default = 0.999, help = "Exponential decay rate of second moment \
                                                                                            for discriminator model optimizer")
    parser.add_argument('--lr_combined', type = float, default = 0.00001, help = "Learning rate for combined model")
    parser.add_argument('--b1_combined', type = float, default = 0.9, help = "Exponential decay rate of first moment \
                                                                                             for combined model optimizer")
    parser.add_argument('--b2_combined', type = float, default = 0.999, help = "Exponential decay rate of second moment \
                                                                                            for combined model optimizer")
    parser.add_argument('--classifier_loss_weight', type = float, default = 1, help = "Classifier loss weight")
    parser.add_argument('--discriminator_loss_weight', type = float, default = 4, help = "Discriminator loss weight")
    parser.add_argument('--batch_size', type = int, default = 32, help = "Batch size for training")
    parser.add_argument('--test_interval', type = int, default = 3, help = "Gap between two successive test phases")
    parser.add_argument('--num_iterations', type = int, default = 12000, help = "Number of iterations")
    parser.add_argument('--snapshot_interval', type = int, default = 500, help = "Minimum gap between saving outputs")
    parser.add_argument('--output_dir', type = str, default = 'Models', help = "Directory for saving outputs")
    args = parser.parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(list(np.arange(args.number_of_gpus))).strip('[]')

    # Initialize parameters
    param = {}
    param["number_of_gpus"] = args.number_of_gpus
    param["network_name"] = args.network_name
    param["inp_dims"] = [224, 224, 3]
    param["num_iterations"] = args.num_iterations
    param["lr_classifier"] = args.lr_classifier
    param["b1_classifier"] = args.b1_classifier
    param["b2_classifier"] = args.b2_classifier    
    param["lr_discriminator"] = args.lr_discriminator
    param["b1_discriminator"] = args.b1_discriminator
    param["b2_discriminator"] = args.b2_discriminator
    param["lr_combined"] = args.lr_combined
    param["b1_combined"] = args.b1_combined
    param["b2_combined"] = args.b2_combined        
    param["batch_size"] = int(args.batch_size/2)
    param["class_loss_weight"] = args.classifier_loss_weight
    param["dis_loss_weight"] = args.discriminator_loss_weight    
    param["drop_classifier"] = args.dropout_classifier
    param["drop_discriminator"] = args.dropout_discriminator
    param["test_interval"] = args.test_interval
    param["source_path"] = os.path.join("Data", args.dataset_name, args.source_path)
    param["target_path"] = os.path.join("Data", args.dataset_name, args.target_path)
    param["snapshot_interval"] = args.snapshot_interval
    param["output_path"] = os.path.join("Snapshot", args.output_dir)

    # Create directory for saving models and log files
    if not os.path.exists(param["output_path"]):
        os.mkdir(param["output_path"])
    
    # Load source and target data
    param["source_data"], param["source_label"] = data_loader(param["source_path"], param["inp_dims"])
    param["target_data"], param["target_label"] = data_loader(param["target_path"], param["inp_dims"])

    # Encode labels into one-hot format
    param["source_label"], param["target_label"] = one_hot_encoding(param)

    # Train data
    train(param)


