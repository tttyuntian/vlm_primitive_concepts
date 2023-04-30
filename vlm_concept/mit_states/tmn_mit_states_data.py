import os
import numpy as np
import torch
from torch.utils.data import Dataset



class MITStatesDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split
        
        # Load metadata
        self.metadata = torch.load(os.path.join(root, "metadata_compositional-split-natural.t7"))
        
        # Load attribute-noun pairs for each split
        all_info, split_info = self.parse_split()
        self.attrs, self.objs, self.pairs = all_info
        self.train_pairs, self.valid_pairs, self.test_pairs = split_info
        
        # Get obj/attr/pair to indices mappings
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.idx2obj = {idx: obj for obj, idx in self.obj2idx.items()}
        self.idx2attr = {idx: attr for attr, idx in self.attr2idx.items()}
        self.idx2pair = {idx: pair for pair, idx in self.pair2idx.items()}
        
        # Get all data
        self.train_data, self.valid_data, self.test_data = self.get_split_info()
        if self.split == "train":
            self.data = self.train_data
        elif self.split == "valid":
            self.data = self.valid_data
        else:
            self.data = self.test_data
        
        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs
    
    def parse_split(self):
        def parse_pairs(pair_path):
            with open(pair_path, "r") as f:
                pairs = f.read().strip().split("\n")
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs
        
        tr_attrs, tr_objs, tr_pairs = parse_pairs(os.path.join(self.root, "compositional-split-natural", "train_pairs.txt"))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(os.path.join(self.root, "compositional-split-natural", "val_pairs.txt"))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(os.path.join(self.root, "compositional-split-natural", "test_pairs.txt"))

        all_attrs = sorted(list(set(tr_attrs + vl_attrs + ts_attrs)))
        all_objs = sorted(list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))
        
        return (all_attrs, all_objs, all_pairs), (tr_pairs, vl_pairs, ts_pairs)
    
    def get_split_info(self):
        train_data, val_data, test_data = [], [], []
        for instance in self.metadata:
            image, attr, obj, settype = instance["image"], instance["attr"], instance["obj"], instance["set"]
            image = image.split("/")[1]  # Get the image name without (attr, obj) folder
            image = os.path.join(" ".join([attr, obj]), image)
            
            if (
                (attr == "NA") or 
                ((attr, obj) not in self.pairs) or 
                (settype == "NA")
            ):
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = {
                "image_path": image, 
                "attr": attr, 
                "obj": obj,
                "pair": (attr, obj),
                "attr_id": self.attr2idx[attr],
                "obj_id": self.obj2idx[obj],
                "pair_id": self.pair2idx[(attr, obj)],
            }
            if settype == "train":
                train_data.append(data_i)
            elif settype == "val":
                val_data.append(data_i)
            else:
                test_data.append(data_i)
                
        return train_data, val_data, test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index = self.sample_indices[index]
        return self.data[index]

def softmax(inputs):
    res = torch.tensor(inputs).float()
    res = res.softmax(dim=-1)
    return res.numpy()

def normalize(inputs):
    res = torch.tensor(inputs).float()
    res /= res.norm(dim=-1, keepdim=True)
    return res.numpy()

def get_gt_primitives(split, data, ):
    """ Get groundtruth primtiive concepts. """
    data_dict = {
        "train": data.train_data,
        "valid": data.valid_data,
        "test": data.test_data,
    }
    split_data = data_dict[split]
    labels_attr = [sample["attr_id"] for sample in split_data]
    labels_obj = [sample["obj_id"] for sample in split_data]
    gt_features_attr = np.zeros((len(split_data), len(data.attrs)))
    gt_features_obj = np.zeros((len(split_data), len(data.objs)))
    gt_features_attr[np.arange(len(labels_attr)), labels_attr] = 1
    gt_features_obj[np.arange(len(labels_obj)), labels_obj] = 1
    gt_features_concat = np.concatenate([gt_features_attr, gt_features_obj], axis=-1)
    return gt_features_concat

def get_precomputed_features(feature, args, is_softmax=False):
    """ Get precomputed CLIP image/pair/attr/obj features """
    data_root = args.precomputed_data_root
    feature_name = "image_features" if feature=="image" else f"{feature}_activations"
    feature_train = np.load(os.path.join(data_root, f"{feature_name}_train.npy"))
    feature_valid = np.load(os.path.join(data_root, f"{feature_name}_valid.npy"))
    feature_test = np.load(os.path.join(data_root, f"{feature_name}_test.npy"))
    if is_softmax:
        feature_train = softmax(feature_train)
        feature_valid = softmax(feature_valid)
        feature_test = softmax(feature_test)
    return feature_train, feature_valid, feature_test
