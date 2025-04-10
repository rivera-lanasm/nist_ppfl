import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle as pkl
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary


from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import PyTorchClassifier

# from attack_targets.cnn.model import Net


# ====================================================
# Data Loading Functions
# ====================================================

def load_model(model_path: Path,
               num_data_features: int,
            #    model_type: str,
               model_class):
    # # import model class based on model type
    # if model_type == 'cnn':
    #     from attack_targets.cnn.model import Net
    # elif model_type == 'dpcnn':
    #     from attack_targets.dpcnn10.model import Net
    # init model
    # model_class = Net
    # create model instance
    model = model_class(num_data_features) 
    # load model weights
    model.load_state_dict(torch.load(model_path)) 
    model.eval()
    return model

def load_data(data_path: Path):
    data = pkl.load(open(data_path, "rb"))
    features = len(data[0]) - 1
    # get feature (one-hot encoded gene variants) columns
    x = data[:, :features]  
    # get label/class (seed coat color) column
    y = data[:, features] 
    return x, y

def load_hyperparameters(hyperparameters_path: Path):
    with open(hyperparameters_path, 'r') as f:
        hyperparameters = json.load(f)
    return hyperparameters



# ====================================================
# LOAD DATA PATHS
# ====================================================

def load_path_set(privacy_type, client_id, model_type):

    # Client model directory path
    model_dir = Path(f'problem1/attack_targets/{privacy_type}/client_{client_id}')
    # Client model (.torch) path
    model_path = Path(model_dir, f'{model_type}_{client_id}.torch')
    # Path to relevant records data file for the client model
    relevant_data_path = Path(model_dir, f'{model_type}_{client_id}_relevant_records.dat')
    # Path to external records data file for the client model
    external_data_path = Path(model_dir, f'{model_type}_{client_id}_external_records.dat')
    # Path to challenge records data file for the client model
    # challenge_data_path = Path(model_dir.parent, f'{privacy_type}_challenge_records.dat')
    # Path to hyperparameters file for the client model
    hyperparameters_path = Path(model_dir, f'{model_type}_{client_id}_hyperparameters.json')

    return model_path, relevant_data_path, external_data_path, hyperparameters_path


# ====================================================
# LOAD MODEL
# ====================================================

def build_attack_model(task_model, num_data_features, hyperparameters_path):

    # These are hyperparameters used for training the client model
    hyperparams = load_hyperparameters(hyperparameters_path)

    # Define loss and other required hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer_name = hyperparams['optimizer']
    learning_rate = hyperparams['learning rate']
    weight_decay = hyperparams['weight decay']
    num_classes = hyperparams['total classes']

    # Wrap client model in the ART PyTorch classifier

    # Select optimizer
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(task_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adamax(task_model.parameters(), lr=learning_rate,  weight_decay=weight_decay)

    # Wrap client model in PyTorchClassifier
    classifier = PyTorchClassifier(
        model=task_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape= (num_data_features,), #(rel_x.shape[1],),
        nb_classes=num_classes
        )

    # Membership Inference Attack model hyperparameters
    attack_model_type = 'nn'
    attack_model_epochs = 100
    attack_model_batch_size = 100
    attack_model_learning_rate = 0.03

    # Create Membership Inference Black Box attack object
    attack_model = MembershipInferenceBlackBox(
        # this is an initialized version of Client N's classification model
        classifier,
        # attack model will be a nn
        attack_model_type=attack_model_type,
        # 100
        nn_model_epochs=attack_model_epochs,
        # 100
        nn_model_batch_size=attack_model_batch_size,
        # 0.03
        nn_model_learning_rate=attack_model_learning_rate
        )
    return attack_model

# ====================================================
# EVALUATE MODEL
# ====================================================

def eval_model(rel_x_preds, ext_x_preds, rel_y, ext_y, attack_model, rel_x, ext_x, challenge_x, challenge_y):
    # Train attack model by passing relevant set as x and external set as test_x
    attack_model.fit(
        # features from relevant records 
        x=rel_x,
        # outcome (soybean class) labels from relevant records
        y=rel_y,
        # features from external records
        test_x=ext_x,
        # outcome (soybean class) labels from external records
        test_y=ext_y,
        # prediction (logits) from pre trained classifier (for a given client) on relevant records 
        pred=rel_x_preds,
        # prediction (logits) from pre trained classifier (for a given client) on external records 
        test_pred=ext_x_preds
        )

    # evaluate challenge records 
    challenge_pred = attack_model.infer(x=challenge_x, y=challenge_y, probabilities=True)

    # pd.Series(challenge_pred.reshape(1,-1)[0]).hist()

    return challenge_pred
