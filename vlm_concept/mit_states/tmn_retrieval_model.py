import argparse
import random
import os
import json
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from .tmn_mit_states_data import MITStatesDataset, get_precomputed_features, normalize, get_gt_primitives
from ..utils.general_tools import set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"



def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--feature", type=str, default="pair",
                        help="['primitive', 'pair', 'all', 'gt_primitive']")
    parser.add_argument("--model", type=str, default="tmn")
    parser.add_argument("--is_open_world", action="store_true")
    parser.add_argument("--is_limit", action="store_true",
                        help="Whehter only limit to seen pairs (1262), or all pairs (1962).")
    parser.add_argument("--train_warmup", type=int, default=100)
    parser.add_argument("--is_image_projection", action="store_true")
    parser.add_argument("--is_bias", action="store_true")
    parser.add_argument("--seed", type=int, default=1123)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=100)

    # I/O parameters
    parser.add_argument("--data_root", type=str, default="./data/mit_states")
    parser.add_argument("--emb_root", type=str, default="./data")
    
    # Necessary training parameters
    parser.add_argument("--input_dim", type=int, default=600)
    parser.add_argument("--output_dim", type=int,
                        help="If 0, then default to image dim.")
    parser.add_argument("--emb_init", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--logit_scale", type=float, default=0.07)
    args = parser.parse_args()

    # I/O parameters
    parser.add_argument("--precomputed_data_root", type=str, default=f"./outputs/mit_states/{args.model}_precompute_features")
    output_path, output_name = get_output_path(args)
    parser.add_argument("--output_path", type=str, default=output_path)
    parser.add_argument("--output_name", type=str, default=output_name)
    args = parser.parse_args()
    return args

def get_output_path(args):
    output_name = "{}_retrieval_model_open{}_init{}_imgproj{}_bias{}_{}_Lim{}_N{}_TW{}_B{}_LR{}_WD{}_L{}_O{}".format(
        args.model,
        1 if args.is_open_world else 0,
        1 if args.emb_init else 0,
        1 if args.is_image_projection else 0,
        1 if args.is_bias else 0,
        args.feature,
        1 if args.is_limit else 0, 
        args.num_epochs,
        args.train_warmup,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.logit_scale,
        args.output_dim,
    )
    output_path = f"./outputs/mit_states/{args.model}_retrieval_model/{output_name}"
    os.makedirs("./outputs/mit_states", exist_ok=True)
    os.makedirs(f"./outputs/mit_states/{args.model}_retrieval_model", exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    return output_path, output_name

def get_logger(args):
    os.makedirs("./logs/mit_states", exist_ok=True)
    os.makedirs(f"./logs/mit_states/{args.model}_retrieval_model", exist_ok=True)
    os.makedirs("./logs/mit_states/{}_retrieval_model/{}".format(args.model, args.output_name), exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = "%(asctime)s %(levelname)s: %(message)s", \
            datefmt = "%m/%d %H:%M:%S %p", \
            filename = "./logs/mit_states/{}_retrieval_model/{}/{}.log".format(args.model, args.output_name, args.output_name), \
            filemode = "w"
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def get_seen_unseen_indices(split, data):
    if split == "train":
        split_data = data.train_data
    elif split == "valid":
        split_data = data.valid_data
    elif split == "test":
        split_data = data.test_data
    else:
        raise ValueError(f"No split found: {split}")

    pairs = [(sample["attr"], sample["obj"]) for sample in split_data]
    seen_indices = [
        i for i in range(len(pairs))
        if pairs[i] in data.train_pairs
    ]
    unseen_indices = [
        i for i in range(len(pairs))
        if pairs[i] not in data.train_pairs
    ]
    print(f"seen_indices: {len(seen_indices)} | unseen_indices: {len(unseen_indices)}")
    return seen_indices, unseen_indices

def evaluate(results):
    """ Evaluate predictions and Return metrics. """
    all_preds, seen_preds, unseen_preds = results["all_preds"], results["seen_preds"], results["unseen_preds"]
    all_acc, seen_acc, unseen_acc = np.mean(all_preds), np.mean(seen_preds), np.mean(unseen_preds)    
    return {
        "all_acc": all_acc,
        "seen_acc": seen_acc,
        "unseen_acc": unseen_acc,
        "harmonic_mean": (seen_acc * unseen_acc)**0.5,
        "macro_average_acc": (seen_acc + unseen_acc)*0.5,
    }

def generate_predictions(scores, labels, seen_ids, unseen_ids, seen_mask, data, topk, bias=0.0):
    """ Apply bias and Generate predictions for. """
    def get_predictions(_scores):
        # Get predictions
        _, pair_preds = _scores.topk(topk, dim=1)
        pair_preds = pair_preds[:, :topk].contiguous().view(-1)
        attr_preds = all_pairs[pair_preds][:,0].view(-1, topk)
        obj_preds = all_pairs[pair_preds][:,1].view(-1, topk)
        pair_preds = pair_preds.view(-1, topk)
        return pair_preds, attr_preds, obj_preds
    
    # Get predictions with biases applied
    all_pairs = torch.LongTensor([
        (data.attr2idx[attr], data.obj2idx[obj]) 
        for attr, obj in data.pairs
    ])
    scores = scores.clone()
    mask = seen_mask.repeat(scores.shape[0], 1)
    scores[~mask] += bias
    pair_preds, attr_preds, obj_preds = get_predictions(scores)
    
    # Get predictions for seen/unseen pairs
    all_preds = np.array([label in pair_preds[row_id,:topk] for row_id, label in enumerate(labels)])
    seen_preds = all_preds[seen_ids]
    unseen_preds = all_preds[unseen_ids]
    return {
        "pair_preds": pair_preds,
        "attr_preds": attr_preds,
        "obj_preds": obj_preds,
        "all_preds": all_preds,
        "seen_preds": seen_preds,
        "unseen_preds": unseen_preds,
    }

def get_overall_metrics(features, labels, seen_ids, unseen_ids, seen_mask, data, topk_list=[1]):
    overall_metrics = {}
    for topk in topk_list:
        # Get model"s performance (accuracy) on seen/unseen pairs
        bias = 1e3
        results = generate_predictions(features, labels, seen_ids, unseen_ids, seen_mask, data, topk=topk, bias=bias)
        full_unseen_metrics = evaluate(results)
        all_preds, seen_preds, unseen_preds = results["all_preds"], results["seen_preds"], results["unseen_preds"]

        # Get predicted probability distribution of unseen pairs,
        # and the top K scores of seen pairs in the predicted prob. distribution of unseen pairs
        correct_scores = features[np.arange(len(features)), labels][unseen_ids]
        max_seen_scores = features[unseen_ids][:, seen_mask].topk(topk, dim=1)[0][:,topk-1]

        # Compute biases
        unseen_score_diff = max_seen_scores - correct_scores
        correct_unseen_score_diff = unseen_score_diff[unseen_preds] - 1e-4
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        bias_list = correct_unseen_score_diff[::bias_skip]

        # Get biased predictions and metrics with different biases
        all_metrics = []
        for bias in bias_list:
            results = generate_predictions(features, labels, seen_ids, unseen_ids, seen_mask, data, topk=topk, bias=bias)
            metrics = evaluate(results)
            all_metrics.append(metrics)
        all_metrics.append(full_unseen_metrics)

        # Compute overall metrics
        seen_accs = np.array([metric_dict["seen_acc"] for metric_dict in all_metrics])
        unseen_accs = np.array([metric_dict["unseen_acc"] for metric_dict in all_metrics])
        best_seen_acc = max([metric_dict["seen_acc"] for metric_dict in all_metrics])
        best_unseen_acc = max([metric_dict["unseen_acc"] for metric_dict in all_metrics])
        best_harmonic_mean = max([metric_dict["harmonic_mean"] for metric_dict in all_metrics])
        auc = np.trapz(seen_accs, unseen_accs)
        #print(f"best_seen_acc: {best_seen_acc:6.4f}")
        #print(f"best_unseen_acc: {best_unseen_acc:6.4f}")
        #print(f"best_harmonic_mean: {best_harmonic_mean:6.4f}")
        #print(f"auc: {auc:6.4f}")

        overall_metrics[topk] = {
            "seen_accs": seen_accs.tolist(),
            "unseen_accs": unseen_accs.tolist(),
            "best_seen_acc": best_seen_acc,
            "best_unseen_acc": best_unseen_acc,
            "best_harmonic_mean": best_harmonic_mean,
            "auc": auc,
        }
    return overall_metrics

def load_fasttext_embeddings(vocab, args):
    custom_map = {
        "Faux.Fur": "fake fur",
        "Faux.Leather": "fake leather",
        "Full.grain.leather": "thick leather",
        "Hair.Calf": "hairy leather",
        "Patent.Leather": "shiny leather",
        "Boots.Ankle": "ankle boots",
        "Boots.Knee.High": "kneehigh boots",
        "Boots.Mid-Calf": "midcalf boots",
        "Shoes.Boat.Shoes": "boatshoes",
        "Shoes.Clogs.and.Mules": "clogs shoes",
        "Shoes.Flats": "flats shoes",
        "Shoes.Heels": "heels",
        "Shoes.Loafers": "loafers",
        "Shoes.Oxfords": "oxford shoes",
        "Shoes.Sneakers.and.Athletic.Shoes": "sneakers",
        "traffic_light": "traficlight",
        "trash_can": "trashcan",
        "dry-erase_board" : "dry_erase_board",
        "black_and_white" : "black_white",
        "eiffel_tower" : "tower"
    }
    vocab_lower = [v.lower() for v in vocab]
    vocab = []
    for current in vocab_lower:
        if current in custom_map:
            vocab.append(custom_map[current])
        else:
            vocab.append(current)

    import fasttext.util
    ft = fasttext.load_model(args.emb_root+"/fasttext/cc.en.300.bin")
    embeds = []
    for k in vocab:
        if "_" in k:
            ks = k.split("_")
            emb = np.stack([ft.get_word_vector(it) for it in ks]).mean(axis=0)
        else:
            emb = ft.get_word_vector(k)
        embeds.append(emb)

    embeds = torch.Tensor(np.stack(embeds))
    logger.info("Fasttext Embeddings loaded, total embeddings: {}".format(embeds.size()))
    return embeds

def load_word2vec_embeddings(vocab, args):
    # vocab = [v.lower() for v in vocab]

    from gensim import models
    model = models.KeyedVectors.load_word2vec_format(
        args.emb_root+"/word2vec/GoogleNews-vectors-negative300.bin", binary=True
    )

    custom_map = {
        "Faux.Fur": "fake_fur",
        "Faux.Leather": "fake_leather",
        "Full.grain.leather": "thick_leather",
        "Hair.Calf": "hair_leather",
        "Patent.Leather": "shiny_leather",
        "Boots.Ankle": "ankle_boots",
        "Boots.Knee.High": "knee_high_boots",
        "Boots.Mid-Calf": "midcalf_boots",
        "Shoes.Boat.Shoes": "boat_shoes",
        "Shoes.Clogs.and.Mules": "clogs_shoes",
        "Shoes.Flats": "flats_shoes",
        "Shoes.Heels": "heels",
        "Shoes.Loafers": "loafers",
        "Shoes.Oxfords": "oxford_shoes",
        "Shoes.Sneakers.and.Athletic.Shoes": "sneakers",
        "traffic_light": "traffic_light",
        "trash_can": "trashcan",
        "dry-erase_board" : "dry_erase_board",
        "black_and_white" : "black_white",
        "eiffel_tower" : "tower"
    }

    embeds = []
    for k in vocab:
        if k in custom_map:
            k = custom_map[k]
        if "_" in k and k not in model:
            ks = k.split("_")
            emb = np.stack([model[it] for it in ks]).mean(axis=0)
        else:
            emb = model[k]
        embeds.append(emb)
    embeds = torch.Tensor(np.stack(embeds))
    logger.info("Word2Vec Embeddings loaded, total embeddings: {}".format(embeds.size()))
    return embeds

def get_pretrained_weights(vocab, args):
    embeds1 = load_fasttext_embeddings(vocab, args)
    embeds2 = load_word2vec_embeddings(vocab, args)
    embeds = torch.cat([embeds1, embeds2], dim = 1)
    logger.info("Combined embeddings are ".format(embeds.size()))
    return embeds

class Precomputed_MITStatesDataset(Dataset):
    def __init__(self, split, feature, data, args, is_limit=True):
        # Load precomputed features with temporary seen_mask
        self.seen_mask = torch.BoolTensor([1 if pair in data.train_pairs else 0 for pair in data.pairs])
        if feature == "primitive":
            attr_actvs_tuple = get_precomputed_features("attr", args, is_softmax=False)
            obj_actvs_tuple = get_precomputed_features("obj", args, is_softmax=False)
            attr_actvs_train, attr_actvs_valid, attr_actvs_test = attr_actvs_tuple
            obj_actvs_train, obj_actvs_valid, obj_actvs_test = obj_actvs_tuple
            image_embs_train = np.concatenate([attr_actvs_train, obj_actvs_train], axis=-1)
            image_embs_valid = np.concatenate([attr_actvs_valid, obj_actvs_valid], axis=-1)
            image_embs_test = np.concatenate([attr_actvs_test, obj_actvs_test], axis=-1)
        elif feature == "pair":
            image_embs_tuple = get_precomputed_features("pair", args, is_softmax=False)
            if (is_limit) and (args.model == "tmn"):
                image_embs_tuple = tuple(t[:, self.seen_mask] for t in image_embs_tuple)
            #image_embs_train, image_embs_valid, image_embs_test = tuple(normalize(t) for t in image_embs_tuple)
            image_embs_train, image_embs_valid, image_embs_test = image_embs_tuple
        elif feature == "all":
            attr_actvs_tuple = get_precomputed_features("attr", args, is_softmax=False)
            obj_actvs_tuple = get_precomputed_features("obj", args, is_softmax=False)
            pair_actvs_tuple = get_precomputed_features("pair", args, is_softmax=False)
            if (is_limit) and (args.model == "tmn"):
                pair_actvs_tuple = tuple(t[:, self.seen_mask] for t in pair_actvs_tuple)
            attr_actvs_train, attr_actvs_valid, attr_actvs_test = attr_actvs_tuple
            obj_actvs_train, obj_actvs_valid, obj_actvs_test = obj_actvs_tuple
            pair_actvs_train, pair_actvs_valid, pair_actvs_test = pair_actvs_tuple
            image_embs_train = np.concatenate([attr_actvs_train, obj_actvs_train, pair_actvs_train], axis=-1)
            image_embs_valid = np.concatenate([attr_actvs_valid, obj_actvs_valid, pair_actvs_valid], axis=-1)
            image_embs_test = np.concatenate([attr_actvs_test, obj_actvs_test, pair_actvs_test], axis=-1)
        elif feature == "gt_primitive":
            image_embs_train = get_gt_primitives("train", data)
            image_embs_valid = get_gt_primitives("valid", data)
            image_embs_test = get_gt_primitives("test", data)
        
        # Prepare labels
        self.limit_pair2idx = self.get_limit_pair2idx(data)
        self.open_world_pair2idx = self.get_open_world_pair2idx(split, data)
        
        labels = None
        if split == "train":
            image_embs = image_embs_train
            if args.is_open_world:
                labels_text = [sample["pair"] for sample in data.valid_data]
                labels = [self.open_world_pair2idx[(attr, obj)] for attr, obj in labels_text]
            else:
                if is_limit:
                    labels_text = [sample["pair"] for sample in data.train_data]
                    labels = [self.limit_pair2idx[(attr, obj)] for attr, obj in labels_text]
                else:
                    labels = [sample["pair_id"] for sample in data.train_data]
        elif split == "valid":
            image_embs = image_embs_valid
            if args.is_open_world:
                labels_text = [sample["pair"] for sample in data.valid_data]
                labels = [self.open_world_pair2idx[(attr, obj)] for attr, obj in labels_text]
            else:
                labels = [sample["pair_id"] for sample in data.valid_data]
        elif split == "test":
            image_embs = image_embs_test
            if args.is_open_world:
                labels_text = [sample["pair"] for sample in data.test_data]
                labels = [self.open_world_pair2idx[(attr, obj)] for attr, obj in labels_text]
            else:
                labels = [sample["pair_id"] for sample in data.test_data]
        
        # Compute seen mask, 1 for training pair, 0 for the others
        if args.is_open_world:
            self.seen_mask = torch.BoolTensor([1 if pair in data.train_pairs else 0 for pair in self.open_world_pair2idx.keys()])
        else:
            self.seen_mask = torch.BoolTensor([1 if pair in data.train_pairs else 0 for pair in data.pairs])
        
        self.image_embs = image_embs
        self.labels = np.array(labels)
        self.image_dim = self.image_embs.shape[-1]
        print(f"image_dim: {self.image_dim:6d}")
    
    def get_limit_pair2idx(self, data):
        labels_text = [sample["pair"] for sample in data.train_data]
        limit_pair2idx = {}
        for label in labels_text:
            if label not in limit_pair2idx:
                limit_pair2idx[label] = len(limit_pair2idx)
        return limit_pair2idx
    
    def get_open_world_pair2idx(self, split, data):
        open_world_pair2idx = {}
        for attr in data.attrs:
            for obj in data.objs:
                if (attr, obj) not in open_world_pair2idx:
                    open_world_pair2idx[(attr, obj)] = len(open_world_pair2idx)
        return open_world_pair2idx
    
    def __len__(self):
        return len(self.image_embs)
    
    def __getitem__(self, index):
        image_embs = self.image_embs[index]
        labels = self.labels[index]
        return image_embs, labels
        

class RetrievalModel(nn.Module):
    def __init__(self, data, image_dim, limit_pairs, open_world_pairs, args):
        super(RetrievalModel, self).__init__()
        self.image_dim = image_dim
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim if args.output_dim > 0 else self.image_dim
        self.args = args
        
        self.obj2idx, self.attr2idx, self.pair2idx = data.obj2idx, data.attr2idx, data.pair2idx
        self.train_pairs, self.valid_pairs, self.test_pairs = data.train_pairs, data.valid_pairs, data.test_pairs
        self.limit_pairs = limit_pairs
        self.open_world_pairs = open_world_pairs
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.logit_scale))
        self.attr_encoder = nn.Embedding(len(data.attrs), self.input_dim)
        self.obj_encoder = nn.Embedding(len(data.objs), self.input_dim)
        self.text_projection = nn.Linear(self.input_dim*2, self.output_dim, bias=args.is_bias)
        
        if self.args.is_image_projection:
            self.image_projection = nn.Linear(self.image_dim, self.output_dim, bias=args.is_bias)
    
    def get_limit_pair_inputs(self):
        attr_inputs, obj_inputs = [],[]
        for attr, obj in self.limit_pairs:
            attr_id = self.attr2idx[attr]
            obj_id = self.obj2idx[obj]
            attr_inputs.append(attr_id)
            obj_inputs.append(obj_id)
        attr_inputs = torch.LongTensor(attr_inputs).to(device)
        obj_inputs = torch.LongTensor(obj_inputs).to(device)
        return attr_inputs, obj_inputs

    def get_all_pair_inputs(self):
        attr_inputs, obj_inputs = [],[]
        for attr, obj in self.pair2idx.keys():
            attr_id = self.attr2idx[attr]
            obj_id = self.obj2idx[obj]
            attr_inputs.append(attr_id)
            obj_inputs.append(obj_id)
        attr_inputs = torch.LongTensor(attr_inputs).to(device)
        obj_inputs = torch.LongTensor(obj_inputs).to(device)
        return attr_inputs, obj_inputs
    
    def forward(self, image_embs, pair_labels=None, is_train=True):
        # image_embs refers to image embeddings or concept representations
        #attrs, objs = self.get_all_pair_inputs()
        attrs, objs = self.get_limit_pair_inputs() if is_train else self.get_all_pair_inputs()
        attr_embs = self.attr_encoder(attrs)
        obj_embs = self.obj_encoder(objs)
        pair_embs = torch.cat([attr_embs, obj_embs], dim=1)
        pair_embs = self.text_projection(pair_embs)
        #pair_embs = F.normalize(pair_embs, dim=1)
        
        if self.args.is_image_projection:
            image_embs = self.image_projection(image_embs.float())
        #image_embs = F.normalize(image_embs, dim=1).float()
        
        logit_scale = self.logit_scale.exp()
        logit_scale = logit_scale if logit_scale<=100.0 else 100.0
        logits = logit_scale * image_embs @ pair_embs.t()
        
        loss = None
        if is_train:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, pair_labels)
        return logits, loss

def main(args):
    # Load dataset
    logger.info("Load dataset...")
    data = MITStatesDataset(root=args.data_root, split="train")  # split can be ignored here
    seen_ids_valid, unseen_ids_valid = get_seen_unseen_indices("valid", data)
    seen_ids_test, unseen_ids_test = get_seen_unseen_indices("test", data)
    train_dataset = Precomputed_MITStatesDataset(split="train", feature=args.feature, data=data, args=args, is_limit=True)
    valid_dataset = Precomputed_MITStatesDataset(split="valid", feature=args.feature, data=data, args=args, is_limit=True)
    test_dataset = Precomputed_MITStatesDataset(split="test", feature=args.feature, data=data, args=args, is_limit=True)
    seen_mask = train_dataset.seen_mask

    # Get retrieval model and optimizer
    logger.info("Load model...")
    limit_pairs = list(train_dataset.limit_pair2idx.keys())
    open_world_pairs = list(valid_dataset.open_world_pair2idx.keys())
    model = RetrievalModel(data, train_dataset.image_dim, limit_pairs, open_world_pairs, args)
    model.to(device)
    if args.emb_init:
        logger.info("Initialize attr/obj encoder with word2vec+fasttext embs...")
        pretrained_weight = get_pretrained_weights(data.attrs, args)
        model.attr_encoder.weight.data.copy_(pretrained_weight)
        pretrained_weight = get_pretrained_weights(data.objs, args)
        model.obj_encoder.weight.data.copy_(pretrained_weight)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Training and early stop
    logger.info("Start training...")
    best_auc = float("-inf")
    best_epoch_id = 0
    best_overall_metrics = {}
    for epoch_id in range(args.num_epochs):
        logger.info("="*70)

        # Train model
        model.train()
        loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=args.batch_size, drop_last=True)
        for iter_id, batch in enumerate(loader):
            image_embs, labels = tuple(t.to(device) for t in batch)
            _, loss = model(image_embs, labels, is_train=True)
            loss.backward()
            optimizer.step()
            model.zero_grad()

            if args.verbose and (iter_id % args.report_step == 0):
                logger.info(f"| Epoch {epoch_id:6d} | iter {iter_id:8d} | train_loss {loss.item():8.5f} |")

        # Early stop and Evaluation
        if epoch_id < args.train_warmup:
            # Training warmup
            continue

        model.eval()
        loader = DataLoader(valid_dataset, shuffle=False, pin_memory=True, batch_size=args.batch_size, drop_last=False)
        features = np.zeros([len(valid_dataset), len(data.pairs)])
        for iter_id, batch in enumerate(loader):
            image_embs, labels = tuple(t.to(device) for t in batch)
            logits, loss = model(image_embs, labels, is_train=False)
            prob = logits.softmax(dim=-1).log()
            features[iter_id*args.batch_size:(iter_id+1)*args.batch_size] = prob.detach().cpu().numpy().copy()
        features = torch.tensor(features)

        labels = valid_dataset.labels
        overall_metrics = get_overall_metrics(features, labels, seen_ids_valid, unseen_ids_valid, seen_mask, data, topk_list=[1,2,3])
        auc1, auc2, auc3 = overall_metrics[1]["auc"], overall_metrics[2]["auc"], overall_metrics[3]["auc"]
        logger.info(f"| Epoch {epoch_id:6d} | valid_auc_1 {auc1:8.5f} | valid_auc_2 {auc2:8.5f} | valid_auc_3 {auc3:8.5f} |")
        if auc1 > best_auc:
            logger.info("*"*70)
            best_auc = auc1
            best_epoch_id = epoch_id
            best_overall_metrics = overall_metrics.copy()
            with open(f"{args.output_path}/valid_metrics.json", "w") as f:
                json.dump(overall_metrics, f)
            torch.save(model.state_dict(), f"{args.output_path}/retrieval_model.ckpt")
            auc1 = best_overall_metrics[1]["auc"]
            logger.info(f"Best auc (topk=1) of {auc1:8.5f} found at Epoch {best_epoch_id}.")
            logger.info(f"Valid split metrics and checkpoint saved to {args.output_path}")
            logger.info("*"*70)

    # Load best checkpoint and test
    logger.info("Training done.")
    logger.info("Load best checkpoint and test...")
    model = RetrievalModel(data, train_dataset.image_dim, limit_pairs, open_world_pairs, args)
    model.load_state_dict(torch.load(f"{args.output_path}/retrieval_model.ckpt"))
    model.to(device)

    model.eval()
    loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=args.batch_size, drop_last=False)
    features = np.zeros([len(test_dataset), len(data.pairs)])
    for iter_id, batch in enumerate(loader):
        image_embs, labels = tuple(t.to(device) for t in batch)
        logits, loss = model(image_embs, labels, is_train=False)
        prob = logits.softmax(dim=-1).log()
        features[iter_id * args.batch_size:(iter_id + 1) * args.batch_size] = prob.detach().cpu().numpy().copy()
    features = torch.tensor(features)
    labels = test_dataset.labels
    overall_metrics = get_overall_metrics(features, labels, seen_ids_test, unseen_ids_test, seen_mask, data, topk_list=[1,2,3])
    auc1, auc2, auc3 = overall_metrics[1]["auc"], overall_metrics[2]["auc"], overall_metrics[3]["auc"]
    logger.info(f"| test_auc_1 {auc1:8.5f} | test_auc_2 {auc2:8.5f} | test_auc_3 {auc3:8.5f} |")
    
    with open(f"{args.output_path}/test_metrics.json", "w") as f:
        json.dump(overall_metrics, f)
    auc1 = best_overall_metrics[1]["auc"]

    logger.info(f"Best auc (topk=1) of {auc1:8.5f} found at Epoch {best_epoch_id}.")
    logger.info(f"Valid/test split metrics and checkpoint saved to {args.output_path}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)
    set_seed(args.seed)

    main(args)
