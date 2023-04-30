import argparse
import os
import logging

import clip
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F

from .tmn_mit_states_data import MITStatesDataset, get_precomputed_features

device = "cuda" if torch.cuda.is_available() else "cpu"



def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=100)
    
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dropout", type=float, default=0.1)
    parser.add_argument("--logit_scale", type=float, default=100.0)
    parser.add_argument("--cls_type", type=str, default="linear")

    parser.add_argument("--data_root", type=str, default="./data/mit_states/")
    parser.add_argument("--precomputed_data_root", type=str, default="./outputs/mit_states/tmn_precompute_features")
    parser.add_argument("--exp_name", type=str, default="tmn_projection_model")
    args = parser.parse_args()

    # I/O variables
    parser.add_argument("--output_dir", type=str, default=f"./outputs/mit_states/{args.exp_name}")
    args = parser.parse_args()
    return args

def get_logger(args):
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs/mit_states", exist_ok=True)
    os.makedirs(f"./logs/mit_states/{args.exp_name}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = "%(asctime)s %(levelname)s: %(message)s", \
            datefmt = "%m/%d %H:%M:%S %p", \
            filename = f"./logs/mit_states/{args.exp_name}/{args.exp_name}.log", \
            filemode = "w"
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

class ProjectionHead(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, cls_type, hidden_dropout=0.1, logit_scale=100.0):
        super(ProjectionHead, self).__init__()
        if cls_type == "linear":
            projection = nn.Linear(input_size, output_size)
        elif cls_type == "mlp":
            projection = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(hidden_dropout),
                nn.Linear(hidden_size, output_size)
            )
        else:
            raise ValueError("Classifier type %s not found" % cls_type)
        self.cls_type = cls_type
        self.projection = projection
        self.logit_scale = logit_scale
    
    def forward(self, input_features):
        input_features = self.projection(input_features)
        return input_features
    
class MITStatesFeatureDataset(Dataset):
    def __init__(self, pair_logits, text_features, labels):
        self.pair_logits = pair_logits
        self.text_features = text_features
        self.labels = labels
    
    def __len__(self):
        return len(self.pair_logits)
    
    def __getitem__(self, index):
        pair_logit = torch.tensor(self.pair_logits[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index])
        return pair_logit, label

def get_clip_text_features(data):
    # Load CLIP
    logger.info("Load CLIP...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Prepare text features
    text_features = []
    text_inputs_pair_raw = [f"this is {attr} {obj}" for attr, obj in data.pairs]
    for i in range(len(text_inputs_pair_raw)):
        text = text_inputs_pair_raw[i]
        text_inputs_pair = torch.cat([clip.tokenize(text)]).to(device)
        text_features_pair = model.encode_text(text_inputs_pair)
        text_features.append(text_features_pair.detach().cpu().numpy())
    text_features = np.concatenate(text_features)
    return text_features

def eval(text_model, image_model, data, args):
    text_model.eval()
    image_model.eval()
    pred_list, true_list = [],[]
    loader = DataLoader(data, shuffle=False, pin_memory=True, batch_size=args.batch_size)
    
    text_features_raw = torch.tensor(data.text_features, dtype=torch.float32).to(device)
    logits_list, text_logits_list, image_logits_list = [],[],[]
    with torch.no_grad():
        for iter_id, batch in enumerate(loader):
            pair_logits, labels = tuple(t.to(device) for t in batch)
            text_features = text_model(text_features_raw)
            image_features = image_model(pair_logits)
            text_logits_list.append(text_features.detach().cpu().numpy())
            image_logits_list.append(image_features.detach().cpu().numpy())

            # Compute logits (cosine similarities)
            #pair_logits /= pair_logits.norm(dim=-1, keepdim=True)
            #text_features /= text_features.norm(dim=-1, keepdim=True)
            #logit_scale = model.logit_scale
            #logits = logit_scale * pair_logits @ text_features.t()
            logits = image_features @ text_features.t()
            logits_list.append(logits.detach().cpu().numpy())

            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            pred_list.append(preds)
            true_list.append(labels.detach().cpu().numpy())
        
        logits_list = np.concatenate(logits_list)
        text_logits_list = np.concatenate(text_logits_list)
        image_logits_list = np.concatenate(image_logits_list)
        pred_list = np.concatenate(pred_list)
        true_list = np.concatenate(true_list)
        accuracy = (pred_list == true_list).sum() / len(true_list)
        return accuracy, pred_list, true_list, logits_list, text_logits_list, image_logits_list

def get_projection_model(input_size, output_size, args):
    projection_model = ProjectionHead(
        input_size = input_size, 
        output_size = output_size,
        hidden_size = output_size,
        cls_type = args.cls_type,
        hidden_dropout = args.hidden_dropout,
        logit_scale = args.logit_scale,
    )
    if torch.cuda.device_count() > 1:
        logger(f"Let's use {torch.cuda.device_count()} GPUs.")
        projection_model = nn.DataParallel(projection_model)
    projection_model.to(device)
    return projection_model

def get_normalized_features(input_features):
    mu = np.mean(input_features, axis=0)
    std = np.std(input_features, axis=0)
    norm_features = (input_features - mu) / std
    return norm_features

def main(args):
    # Compute CLIP text features
    logger.info("Compute CLIP text features..")
    data = MITStatesDataset(root=args.data_root, split="train")  # Any split would work
    text_features_pair = get_clip_text_features(data)
    
    # Preprocess data
    logger.info("Preprocessing data..")
    pair_actvs = get_precomputed_features("pair", args, is_softmax=False)
    #pair_logits_train, pair_logits_valid, pair_logits_test = tuple(actv / 100 for actv in pair_actvs)
    pair_logits_train, pair_logits_valid, pair_logits_test = tuple(
        get_normalized_features(actv / 100) for actv in pair_actvs
    )

    labels_train = [sample["pair_id"] for sample in data.train_data]
    labels_valid = [sample["pair_id"] for sample in data.valid_data]
    labels_test = [sample["pair_id"] for sample in data.test_data]
    
    train_dataset = MITStatesFeatureDataset(pair_logits_train, text_features_pair, labels_train)
    valid_dataset = MITStatesFeatureDataset(pair_logits_valid, text_features_pair, labels_valid)
    test_dataset = MITStatesFeatureDataset(pair_logits_test, text_features_pair, labels_test)
    logger.info(f"train_dataset: {len(train_dataset)}")
    logger.info(f"valid_dataset: {len(valid_dataset)}")
    logger.info(f"test_dataset: {len(test_dataset)}")

    # Configure projection model
    logger.info("Configure projection model..")
    input_size = text_features_pair.shape[1]
    output_size = pair_logits_train.shape[1]
    logger.info(f"input_size: {input_size}")
    logger.info(f"output_size: {output_size}")

    text_projection_model = get_projection_model(input_size, output_size, args)
    image_projection_model = get_projection_model(output_size, output_size, args)
    text_optimizer = Adam(text_projection_model.parameters(), lr=args.learning_rate)
    image_optimizer = Adam(image_projection_model.parameters(), lr=args.learning_rate)
    
    # Start training
    logger.info("Start training..")
    best_accuracy = float("-inf")
    best_valid_id = 0

    text_features_raw = torch.tensor(train_dataset.text_features, dtype=torch.float32).to(device)
    for epoch_id in range(args.num_epochs):
        logger.info("="*70)

        # Training phase
        text_projection_model.train()
        image_projection_model.train()
        loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=args.batch_size, drop_last=True)
        for iter_id, batch in enumerate(loader):
            pair_logits, labels = tuple(t.to(device) for t in batch)
            text_features = text_projection_model(text_features_raw)
            image_features = image_projection_model(pair_logits)
            
            # Compute logits (cosine similarities)
            #pair_logits = pair_logits / pair_logits.norm(dim=-1, keepdim=True)
            #text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            #logit_scale = text_projection_model.logit_scale
            #logits = logit_scale * pair_logits @ text_features.t()
            logits = image_features @ text_features.t()
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            text_optimizer.step()
            image_optimizer.step()
            text_projection_model.zero_grad()
            image_projection_model.zero_grad()
            
            if args.verbose and (iter_id % args.report_step == 0):
                logger.info(f"| Epoch {epoch_id:6d} | iter {iter_id:8d} | train_loss {loss.item():8.5f} |")
        
        accuracy, _, _, _, _, _= eval(text_projection_model, image_projection_model, train_dataset, args)
        logger.info(f"training acc: {float(accuracy):8.6f}")

        # Early stopping and Evaluation
        accuracy, pred_list, true_list, logits_list, text_logits_list, image_logits_list = eval(
            text_projection_model, image_projection_model, valid_dataset, args
        )
        logger.info(f"Valid accuracy: {float(accuracy):8.6f}")
        if accuracy > best_accuracy:
            logger.info("*"*70)
            best_accuracy = accuracy
            best_epoch_id = epoch_id
            logger.info(f"Epoch {epoch_id}, best model found with accuracy {float(accuracy):8.5f}")
            
            # Output logits and model
            os.makedirs(args.output_dir, exist_ok=True)
            np.save(os.path.join(args.output_dir, "logits_valid.npy"), logits_list)
            np.save(os.path.join(args.output_dir, "text_logits_valid.npy"), text_logits_list)
            np.save(os.path.join(args.output_dir, "image_logits_valid.npy"), image_logits_list)
            np.save(os.path.join(args.output_dir, "preds_list_valid.npy"), pred_list)
            np.save(os.path.join(args.output_dir, "true_list_valid.npy"), true_list)
        
            text_projection_model.eval()
            image_projection_model.eval()
            text_model_output_path = os.path.join(args.output_dir, "text_projection_model.pt")
            image_model_output_path = os.path.join(args.output_dir, "image_projection_model.pt")
            torch.save(text_projection_model.state_dict(), text_model_output_path)
            torch.save(image_projection_model.state_dict(), image_model_output_path)
            logger.info(f"*"*70)
        logger.info("="*70)

    logger.info(f"Training done")
    logger.info(f"Best accuracy {best_accuracy:8.5f} at Epoch {best_epoch_id}")
    logger.info(f"Logits and model are saved to {args.output_dir}")

    # Load best model to run on test split
    logger.info("Read best model")
    text_projection_model = get_projection_model(input_size, output_size, args)
    image_projection_model = get_projection_model(output_size, output_size, args)

    text_model_path = os.path.join(args.output_dir, "text_projection_model.pt")
    image_model_path = os.path.join(args.output_dir, "image_projection_model.pt")
    text_projection_model.load_state_dict(torch.load(text_model_path))
    image_projection_model.load_state_dict(torch.load(image_model_path))

    logger.info("Run on test split")
    accuracy, pred_list, true_list, logits_list, text_logits_list, image_logits_list = eval(
        text_projection_model, image_projection_model, valid_dataset, args
    )
    logger.info(f"Test accuracy: {float(accuracy):8.6f}")
    np.save(os.path.join(args.output_dir, "logits_test.npy"), logits_list)
    np.save(os.path.join(args.output_dir, "text_logits_test.npy"), text_logits_list)
    np.save(os.path.join(args.output_dir, "image_logits_test.npy"), image_logits_list)
    np.save(os.path.join(args.output_dir, "preds_list_test.npy"), pred_list)
    np.save(os.path.join(args.output_dir, "true_list_test.npy"), true_list)
    
    logger.info(f"Model and logits output to {args.output_dir}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)

    main(args)
