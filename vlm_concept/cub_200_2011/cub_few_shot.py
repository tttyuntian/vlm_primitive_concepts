import os
from typing import DefaultDict
import numpy as np
import pandas as pd
import torch
import clip
import argparse
import pdb

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn import svm
from torch.utils.data import DataLoader
from PIL import Image
from scipy import stats
import copy
import random


def get_few_shot_idx(train_y, K, classes, classes_idx):
    '''
    inputs: 
        train_y: training labels from 0-199
        K: number of training examples per class (k-shot)
        classes: class labels chosen for task (n-way)
    
    outputs:
        training indices for n-way k-shot
    '''

    idx = []
    for cl in classes:
        id = classes_idx[cl]
        random.shuffle(id)
        idx += id[:K]

    return idx


# RETURNS IMAGE FEATURES FROM PRETRAINED VL MODEL
def get_features(model, dataset):
    all_features = []
    
    with torch.no_grad():
        for images in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.cuda())
            all_features.append(features)
    return torch.cat(all_features)

# RETURNS TEXT FEATURES FROM PRETRAINED VL MODEL
def get_text_features(model, dataset):
    '''
    inputs: 
        model: pretrained VL model
    outputs:
        text features
    '''

    all_features = []
    
    with torch.no_grad():
        for text in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_text(text.cuda())
            all_features.append(features)
    return torch.cat(all_features)


# RETURNS DATA DICTIONARY AND LABELS
def load_data(args, split):
    # Load image data
    images = pd.read_csv(
        os.path.join(args.data_root, "CUB_200_2011", "images.txt"),
        sep=" ", names=["image_id", "filepath"],
    )
    image_class_labels = pd.read_csv(
        os.path.join(args.data_root, "CUB_200_2011", "image_class_labels.txt"),
        sep=" ", names=["image_id", "class_id"],
    )
    train_test_split = pd.read_csv(
        os.path.join(args.data_root, "CUB_200_2011", "train_test_split.txt"),
        sep=" ", names=["image_id", "is_training_image"],
    )
    classes = pd.read_csv(
        os.path.join(args.data_root, "CUB_200_2011", "classes.txt"),
        sep=" ", names=["class_id", "class_name"],
    )
    data = images.merge(image_class_labels, on="image_id")
    data = data.merge(train_test_split, on="image_id")
    data = data.merge(classes, on="class_id")

    # create groundtruth attribute labels
    gt_attr = np.zeros((11788, 312))
    attr_file = open(os.path.join(args.data_root, "CUB_200_2011", "attributes", "image_attribute_labels.txt"))
    
    for l in attr_file.readlines():
        s = l.split(" ")
        gt_attr[int(s[0])-1, int(s[1])-1] = int(s[2])

    new_gt_attr = copy.copy(gt_attr)

    all_y_labels = np.array(data["class_id"]).astype(np.int)
    all_y_labels -=1

    class_attr = np.zeros((200, 312))

    for cl in range(200):
        rows = np.squeeze(np.argwhere(all_y_labels==cl))
        class_data = gt_attr[rows]

        for at in range(312):
            maj_att = stats.mode(class_data[:,at])

            for row in rows:
                new_gt_attr[row, at] = maj_att[0]
                class_attr[cl, at] = maj_att[0]


    # Get data split
    if split=="train":
        new_gt_attr = new_gt_attr[data.is_training_image==1]
        gt_attr = gt_attr[data.is_training_image==1]
        data = data[data.is_training_image==1]
        

    elif split=="valid":
        new_gt_attr = new_gt_attr[data.is_training_image==0]
        gt_attr = gt_attr[data.is_training_image==0]
        data = data[data.is_training_image==0]

    elif split=="all":
        data = data
        new_gt_attr = new_gt_attr

    data["class_name"] = [class_name.split(".")[1].lower().replace("_", " ") for class_name in data.class_name]
    data["image_id"] = data["image_id"].astype(int)
    data.sort_values(by="image_id")

    data["class_id"] = data["class_id"].astype(int)

    y_labels = np.array(data["class_id"]).astype(np.int)
    y_labels -=1

    ind = []
    class_ind = DefaultDict(list)

    for i, ex in enumerate(new_gt_attr):
        if np.mean(np.where(class_attr[y_labels[i]]==ex, 1, 0)) > 0.98:
            ind.append(i)

    for i, yl in enumerate(y_labels.tolist()):
        class_ind[yl].append(i)

    return data, y_labels, torch.FloatTensor(new_gt_attr), gt_attr, classes, class_ind, class_attr


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="../data/cub/")
    parser.add_argument('-n','--nlist', nargs='+', type=int, default=[5])
    parser.add_argument('-k','--klist', nargs='+', type=int, default=[5])
    parser.add_argument('-a', '--activations_root', type=str, default="clip_cub_features/")
    parser.add_argument('-e', '--num_exp', type=int, default=600)
    args = parser.parse_args()
    print(args)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    model = model.to(device)
    print(device)

    classes = pd.read_csv(
        os.path.join(args.data_root, "CUB_200_2011", "classes.txt"),
        sep=" ", names=["class_id", "class_name"],
    )

    # LOAD TRAIN IMAGES
    train_data, train_y, train_attr_gt, tr_gt2, classes, class_ind, class_attr = load_data(args, split="train")


    # LOAD TEST IMAGES
    test_data, test_y, test_attr_gt, t_gt2, classes, _, _ = load_data(args, split="valid")

    # primitive concept activations
    tr_att_activations = torch.Tensor(np.load(args.activations_root + "attribute_activations_train.npy"))
    t_att_activations = torch.Tensor(np.load(args.activations_root + "attribute_activations_valid.npy"))

    # all activations
    # tr_att_activations = torch.cat((tr_class_activations, tr_att_activations), 1)
    # t_att_activations = torch.cat((t_class_activations, t_att_activations), 1)
    

    n = args.nlist
    k = args.klist

    print("N", "K", "Prim,", "Interv,", "IntervX,", "GT")
    # pdb.set_trace()
    for N in n:
        for K in k:

            results = []

            for exp in tqdm(range(args.num_exp)):

                curr_result = []

                # GET N CLASS LABELS
                classes = np.arange(200)
                np.random.shuffle(classes)
                classes = classes[:N]
                classes.sort()


                # GET K TRAINING IMAGES PER CLASS
                idx = get_few_shot_idx(train_y, K, classes, class_ind)
                train_y2 = train_y[idx]
                tr_prim = tr_att_activations[idx].numpy()
                tr_prim_gt = train_attr_gt[idx]

                # GET TESTING IMAGES FROM N CLASSES
                t_idx = np.argwhere(np.in1d(test_y, classes)).flatten()
                test_y2 = test_y[t_idx]
                t_prim = t_att_activations[t_idx].numpy()
                t_prim_gt = test_attr_gt[t_idx]
                

                # ConceptCLIP - Primitive (Logistic Regression)
                classifier = LogisticRegression()
                classifier.fit(tr_prim, train_y2)
                score = classifier.score(t_prim, test_y2)
                intervention_score = classifier.score(t_prim_gt, test_y2)
                gt_x = np.where(t_prim_gt==1, 1, t_prim)
                x_interv_score = classifier.score(gt_x, test_y2) 

                gt_classifier = LogisticRegression()
                gt_classifier.fit(tr_prim_gt, train_y2)
                gt_score = gt_classifier.score(t_prim_gt, test_y2)

                results.append([score, intervention_score, x_interv_score, gt_score])

            results = np.array(results)
            print(N, K, np.around(np.mean(results, axis=0), 3))

if __name__ == "__main__":
    main()