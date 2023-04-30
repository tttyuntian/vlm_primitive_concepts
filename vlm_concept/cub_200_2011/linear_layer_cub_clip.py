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

def get_features(model, dataset):
    all_features = []
    
    with torch.no_grad():
        for images in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.cuda())
            all_features.append(features)
    return torch.cat(all_features)

def get_text_features(model, dataset):
    all_features = []
    
    with torch.no_grad():
        for text in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_text(text.cuda())
            all_features.append(features)
    return torch.cat(all_features)


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


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.logreg = torch.nn.Linear(512, 200).cuda()
        self.text_weights = torch.nn.Linear(1024, 512).cuda()
        # self.weights = torch.nn.Linear(1024, 200).cuda()

    def forward(self, features):
        tr_sim = torch.nn.functional.softmax(100*self.text_weights(features.cuda()), dim=1)
        out = self.logreg(tr_sim)
        # out = self.weights(features)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="./")
    parser.add_argument('-n','--nlist', nargs='+', type=int)
    parser.add_argument('-k','--klist', nargs='+', type=int)
    args = parser.parse_args()
    print(args)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50', device)
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
    train_text_inputs = torch.cat([clip.tokenize("a photo of a " + str(c)) for c in classes["class_name"]])
    
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
    train_features = get_features(model, train_ims)
    test_features = get_features(model, test_ims)
    text_features = get_text_features(model, train_text_inputs)

    # NORMALIZE CLIP FEATURES
    train_feats = train_features.type(torch.FloatTensor).cuda()
    train_feats /= torch.norm(train_feats, dim=-1, keepdim=True)
    test_feats = test_features.type(torch.FloatTensor).cuda()
    test_feats /= torch.norm(test_feats, dim=-1, keepdim=True)
    text_labels = text_features.type(torch.FloatTensor).cpu()
    text_labels /= torch.norm(text_labels, dim=-1, keepdim=True)


    m = Model().cuda()
    loss_fn = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(m.parameters(), lr=0.1)

    for ep in range(100000):

        optimizer.zero_grad()
        outputs = m(train_feats)

        loss = loss_fn(outputs, torch.LongTensor(train_y).cuda())
        loss.backward()
        optimizer.step()

        if not ep % 1000:
            # print(loss)
            outputs = m(test_feats)
            acc = (torch.argmax(outputs, 1) == torch.LongTensor(test_y).cuda()).sum()/len(outputs)
            print(acc.item())



if __name__ == "__main__":
    main()