import os
import numpy as np
import pandas as pd
import torch
import clip
import argparse
import pdb

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from torch.utils.data import DataLoader
from PIL import Image


def get_few_shot_idx(train_y, K, classes):
    '''
    inputs: 
        train_y: training labels from 0-199
        K: number of training examples per class (k-shot)
        classes: class labels chosen for task (n-way)
    
    outputs:
        training indices for n-way k-shot
    '''
    idx = np.arange(6000)
    idx[-6:] -= 6
    idx = idx.reshape((200, -1))
    ignore = [np.random.shuffle(x) for x in idx]
    
    idx = idx[:,:K].flatten()
    
    corr = np.concatenate([[i]*K for i in range(200)])
    y = train_y[idx]
    
    # CORRECT DISTRIBUTION IF NOT UNIFORM
    foo = (y == corr)
    for i, val in enumerate(foo):
        if not val:
            while train_y[idx[i]] < corr[i]:
                idx[i] += 1
            while train_y[idx[i]] > corr[i]:
                idx[i] -= 1

    y = train_y[idx]

    classes_idx = np.argwhere(np.in1d(y, classes)).flatten()
    idx = idx[classes_idx]

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

    # Get data split
    if split=="train":
        data = data[data.is_training_image==1]
    elif split=="valid":
        data = data[data.is_training_image==0]
    elif split=="all":
        data = data
    data["class_name"] = [class_name.split(".")[1].lower().replace("_", " ") for class_name in data.class_name]
    
    # Load attribute data
    image_attribute_labels = pd.read_csv(
        os.path.join(args.data_root, "CUB_200_2011", "attributes", "image_attribute_labels.txt"),
        sep=" ", names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
    )
    attributes = pd.read_csv(
        os.path.join(args.data_root, "CUB_200_2011", "attributes", "attributes.txt"),
        sep=" ", names=["attribute_id", "attribute_name"]
    )
    attributes_info = [attr.split("::") for attr in attributes.attribute_name]
    attributes_info = np.array([[attr.replace("_", " "), label.replace("_", " ")] for attr, label in attributes_info])
    attributes["attribute_template"] = attributes_info[:, 0]
    attributes["attribute_label"] = attributes_info[:, 1]
    attributes = image_attribute_labels.merge(attributes, on="attribute_id")
    unique_attributes = attributes.attribute_template.unique()

    data["image_id"] = data["image_id"].astype(int)
    data.sort_values(by="image_id")

    data["class_id"] = data["class_id"].astype(int)

    y_labels = np.array(data["class_id"]).astype(np.int)
    y_labels -=1

    return data, y_labels, unique_attributes, classes


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="./")
    parser.add_argument('-n','--nlist', nargs='+', type=int, help='<Required> Set flag', required=True)
    parser.add_argument('-k','--klist', nargs='+', type=int, help='<Required> Set flag', required=True)
    parser.add_argument('-a', '--activations_root', type=str, default="./")
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
    train_data, train_y, _, classes = load_data(args, split="train")
    train_ims = []
    for x in train_data["filepath"]:
        im = Image.open(args.data_root + "/CUB_200_2011/images/" + x)
        train_ims.append(np.array(preprocess(im)))
        im.load()
    
    train_ims = np.array(train_ims)
    train_y = np.array(train_y)
    train_ims = torch.Tensor(train_ims).cuda()
    train_text_inputs = torch.cat([clip.tokenize("a photo of a " + str(c) +", a type of fruit") for c in classes["class_name"]])
    
    # LOAD TEST IMAGES
    test_data, test_y, _, classes = load_data(args, split="valid")
    test_ims = []
    for x in test_data["filepath"]:
        im = Image.open(args.data_root + "/CUB_200_2011/images/" + x)
        test_ims.append(np.array(preprocess(im)))
        im.load()
    
    test_ims = np.array(test_ims)
    test_ims = torch.Tensor(test_ims).cuda()

    # GET CLIP FEATURES 
    # train_features = get_features(model, train_ims)
    # test_features = get_features(model, test_ims)
    # text_features = get_text_features(model, train_text_inputs)

    # SAVED CLIP FEATURES
    train_features = torch.Tensor(np.load(args.activations_root + "image_features_train.npy"))
    test_features = torch.Tensor(np.load(args.activations_root + "image_features_valid.npy"))

    # USE SAVED ACTIVATIONS
    # composite concept activations
    tr_class_activations = torch.nn.functional.softmax(100*torch.Tensor(np.load(args.activations_root + "class_activations_train.npy")), dim=1)
    t_class_activations = torch.nn.functional.softmax(100*torch.Tensor(np.load(args.activations_root + "class_activations_valid.npy")), dim=1)

    # primitive concept activations
    tr_att_activations = torch.Tensor(np.load(args.activations_root + "attribute_activations_train.npy"))
    t_att_activations = torch.Tensor(np.load(args.activations_root + "attribute_activations_valid.npy"))

    # all activations
    tr_sim = torch.cat((tr_class_activations, tr_att_activations), 1)
    t_sim = torch.cat((t_class_activations, t_att_activations), 1)
    

    n = args.nlist
    k = args.klist

    for N in n:
        for K in k:

            results = []

            for exp in tqdm(range(600)):

                curr_result = []

                # GET N CLASS LABELS
                classes = np.arange(200)
                np.random.shuffle(classes)
                classes = classes[:N]

                # GET K TRAINING IMAGES PER CLASS
                idx = get_few_shot_idx(train_y, K, classes)
                train_feats = train_features[idx]
                train_y2 = train_y[idx]
                tr_similarity = tr_sim[idx]
                tr_prim = tr_att_activations[idx]
                tr_comp = tr_class_activations[idx]

                # GET TESTING IMAGES FROM N CLASSES
                t_idx = np.argwhere(np.in1d(test_y, classes)).flatten()
                test_feats = test_features[t_idx]
                test_y2 = test_y[t_idx]
                t_similarity = t_sim[t_idx]
                t_prim = t_att_activations[t_idx]
                t_comp = t_class_activations[t_idx]    
                
                # ZERO SHOT ACCURACY
                acc = (torch.argmax(tr_similarity.type(torch.FloatTensor), 1).cuda() == torch.Tensor(train_y2).type(torch.FloatTensor).cuda()).type(torch.float).sum().item()
                acc /= len(tr_similarity)
                curr_result.append(acc)

                t_acc = (torch.argmax(t_similarity.type(torch.FloatTensor), 1).cuda() == torch.Tensor(test_y2).type(torch.FloatTensor).cuda()).type(torch.float).sum().item()
                t_acc /= len(t_similarity)
                curr_result.append(t_acc)

                # ConceptCLIP - Primitive (Logistic Regression)
                classifier = LogisticRegression()
                classifier.fit(tr_prim.cpu().numpy(), train_y2)
                lr_score = classifier.score(t_prim.cpu().numpy(), test_y2)
                curr_result.append(lr_score)

                # ConceptCLIP - Primitive (SVM)
                clf = svm.SVC(kernel='linear') 
                clf.fit(tr_prim, train_y2)
                svm_score = clf.score(t_prim.cpu().numpy(), test_y2)
                curr_result.append(svm_score)

                # ConceptCLIP - Composite (Logistic Regression)
                classifier = LogisticRegression()
                classifier.fit(tr_comp.cpu().numpy(), train_y2)
                lr_score = classifier.score(t_comp.cpu().numpy(), test_y2)
                curr_result.append(lr_score)

                # ConceptCLIP - Composite (SVM)
                clf = svm.SVC(kernel='linear') 
                clf.fit(tr_comp, train_y2)
                svm_score = clf.score(t_comp.cpu().numpy(), test_y2)
                curr_result.append(svm_score)

                # ConceptCLIP - ALL (Logistic Regression)
                classifier = LogisticRegression()
                classifier.fit(tr_similarity.cpu().numpy(), train_y2)
                lr_score = classifier.score(t_similarity.cpu().numpy(), test_y2)
                curr_result.append(lr_score)

                # ConceptCLIP - ALL (SVM)
                clf = svm.SVC(kernel='linear') 
                clf.fit(tr_similarity, train_y2)
                svm_score = clf.score(t_similarity.cpu().numpy(), test_y2)
                curr_result.append(svm_score)
                
                # CLIP Features (Logistic Regression)
                classifier2 = LogisticRegression()
                classifier2.fit(train_feats, train_y2)
                lr_score2 = classifier2.score(test_feats.cpu().numpy(), test_y2)
                curr_result.append(lr_score2)

                # CLIP Features (SVM)
                clf2 = svm.SVC(kernel='linear') 
                clf2.fit(train_feats, train_y2)
                svm2_score = clf2.score(test_feats.cpu().numpy(), test_y2)
                curr_result.append(svm2_score)

                results.append(curr_result)

            results = np.array(results)
            print(N, K, np.mean(results, axis=0), flush=True)


if __name__ == "__main__":
    main()