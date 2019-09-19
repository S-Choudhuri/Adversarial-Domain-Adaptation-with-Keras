import os
import argparse
import random
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.layers import Input
from keras.optimizers import Adam
import model
import loss

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
    classes = (np.concatenate((s_label, t_label),axis=0))
    num_classes = np.max(classes)
    if 0 in classes:
            num_classes = num_classes+1
    s_label = to_categorical(s_label, num_classes=num_classes)
    t_label = to_categorical(t_label, num_classes=num_classes)
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
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr

def train(param):
    models = {}
    inp = Input(shape=(param["inp_dims"]))
    embedding = model.build_embedding(inp)
    classifier = model.build_classifier(param, embedding)
    discriminator = model.build_discriminator(embedding)

    models["combined_classifier"] = model.build_combined_classifier(inp, classifier)
    models["combined_classifier"].compile(optimizer="Adam",loss='categorical_crossentropy', metrics=['accuracy'])

    models["combined_discriminator"] = model.build_combined_discriminator(inp, discriminator)
    models["combined_discriminator"].compile(optimizer="Adam",loss='binary_crossentropy', metrics=['accuracy'])

    models["combined_model"] = model.build_combined_model(inp, [classifier, discriminator])
    models["combined_model"].compile(optimizer="Adam",loss={'c_dense2': 'categorical_crossentropy', 'd_dense2': \
                       'binary_crossentropy'}, loss_weights={'c_dense2': 1, 'd_dense2': 2}, metrics=['accuracy'])
    
    y_adversarial_1 = np.array(([1] * param["batch_size"] + [0] * param["batch_size"]))
    sample_weights_adversarial = np.ones((param["batch_size"] * 2,))
    S_batches = batch_generator([Xs, to_categorical(ys)], param["batch_size"])
    T_batches = batch_generator([Xt, np.zeros(shape = (len(Xt),2))], param["batch_size"])

    for i in range(param["num_iterations"]):
        y_adversarial_2 = to_categorical(np.array(([0] * batch_size + [1] * batch_size)))
        X0, y0 = next(S_batches)
        X1, y1 = next(T_batches)
        X_adv = np.concatenate([X0, X1])
        y_class = np.concatenate([y0, np.zeros_like(y0)])
        adv_weights = []
        for layer in model.layers:
            if (layer.name.startswith("do")):
                adv_weights.append(layer.get_weights())
        we = domain_classification_model.predict(X0)
            we = 1-we[:,0]
            we = we.tolist()
            sample_weights_class = np.array((we + [0] * batch_size))            
            stats = model.train_on_batch(X_adv, [y_class, y_adversarial_1],
                                     sample_weight=[sample_weights_class, sample_weights_adversarial])            
            k = 0
            for layer in model.layers:
                if (layer.name.startswith("do")):                    
                    layer.set_weights(adv_weights[k])
                    k += 1
            class_weights = []        
            for layer in model.layers:
                if (not layer.name.startswith("do")):
                    class_weights.append(layer.get_weights())            
            stats2 = domain_classification_model.train_on_batch(X_adv, [y_adversarial_2])
            k = 0
            for layer in model.layers:
                if (not layer.name.startswith("do")):
                    layer.set_weights(class_weights[k])
                    k += 1

        if ((i + 1) % 1000 == 0):
            # print(i, stats)
            y_test_hat_t = source_classification_model.predict(Xt).argmax(1)
            y_test_hat_s = source_classification_model.predict(Xs).argmax(1)
            print("Iteration %d, source accuracy =  %.3f, target accuracy = %.3f"%(i, accuracy_score(ys, y_test_hat_s), accuracy_score(yt, y_test_hat_t)))


if __name__ == "__main__":
    # Read parameter values from the console
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="GPU id to run")
    parser.add_argument('--network_name', type=str, default='ResNet50', help="Name of the feature extractor network; ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--dataset_name', type=str, default='office', help="Name of the source dataset")
    parser.add_argument('--source_path', type=str, default='Data/Office/amazon_10_list.txt', help="Path to source dataset")
    parser.add_argument('--target_path', type=str, default='Data/Office/webcam_10_list.txt', help="Path to target dataset")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
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
    param["batch_size"] = args.batch_size
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

    # Load source and target data
    param["source_data"], param["source_label"] = data_loader(param["source_path"], param["inp_dims"])
    param["target_data"], param["target_label"] = data_loader(param["target_path"], param["inp_dims"])

    # Encode labels into one-hot format
    param["source_label"], param["target_label"] = one_hot_encoding(param)

    # Train data
    train(param)


