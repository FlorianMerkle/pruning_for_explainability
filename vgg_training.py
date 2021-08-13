import torch
import torch.nn as nn
from torchvision import models, datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse

# my own functions
from helpers import set_device, load_run_config, imagenette_dataloader, disable_params_grad, get_params_grad, create_optimizer, train_model, \
    test_model, save_model, save_to_json

### Sources
# PyTorch Tutorial: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# Theory: https://cs231n.github.io/transfer-learning/, https://ruder.io/transfer-learning/
# Notes: Due to the smaller size and similarity of Imagenette to Imagenet, it might make sense to only do
#       transfer learning on the fc-layers at the end (prevent overfitting).

### Features:
# Optimizer: weight_decay (L2 penalty to make parameters smaller), LR Scheduler (adapt lr dynamically)
# Test loading of a saved model


def main():
    parser = argparse.ArgumentParser(description='Start skript for VGG16 Training runs')
    parser.add_argument('config_file_path', type=str, help='Specify the path to the config file')
    parser.add_argument('desired_run', type=str, help='Specify which run of the config file should be executed')

    arguments = parser.parse_args()

    config_path = arguments.config_file_path
    desired_run = arguments.desired_run

    # load run-config
    run_configuration = load_run_config(config_path, desired_run)

    # start vgg run
    execute_vgg_run(run_configuration, desired_run)


def execute_vgg_run(run_configuration, desired_run):
    # raise Exception('Can I load an already saved model? Test it! Same accuracy? What about require_grad=True for disabled layers???')
    # get parameters
    batch_size = run_configuration.get('batch_size')
    epochs = run_configuration.get('epochs')
    pretrained = run_configuration.get('pretrained')
    vgg_no_train_parts = run_configuration.get('vgg_no_train_parts')
    optimizer_str = run_configuration.get('optimizer')
    learning_rate = run_configuration.get('learning_rate')
    momentum = run_configuration.get('momentum')
    model_path = run_configuration.get('model_path')
    results_path = run_configuration.get('results_path')

    # set device
    device = set_device()


    # load dataset and model
    if pretrained:
        print('Loading pretrained VGG-16; val-set from test-set')
    else:
        print('Loading randomly initialized VGG-16; val-set from test-set')

    trainloader, validloader, testloader = imagenette_dataloader(batch_size)
    vgg16_model = models.vgg16(pretrained=pretrained)
    # move model to device
    vgg16_model = vgg16_model.to(device)

    # feature extraction? If yes, disable grads of desired parts
    # disable_params_grad(vgg16_model, vgg_no_train_parts)

    # in_features of last linear layer
    required_in_features = vgg16_model.classifier[-1].in_features

    # reinitialize last linear layer for the correct amount of classes (out_features)
    num_classes = 10
    vgg16_model.classifier[-1] = nn.Linear(in_features=required_in_features, out_features=num_classes)
    # params_w_grad = get_params_grad(vgg16_model)

    # move model to device
    vgg16_model = vgg16_model.to(device)

    # specify optimizer
    optimizer = create_optimizer(list(vgg16_model.parameters()), optimizer_str, learning_rate, momentum)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # start training
    best_model, epoch, train_acc, val_acc, val_acc_history, time_elapsed = train_model(vgg16_model, trainloader, validloader, epochs, optimizer, criterion, device)

    # Test Model on TestSet
    top1acc, top5acc = test_model(best_model, testloader, device)

    # save model
    path = save_model(best_model, model_path)

    # save outputs of each run in results.json: basically the best model with certain parameters
    results = {
        'name_of_run': desired_run,
        'best_model_path': path,
        'was_it_pretrained': pretrained,
        'training_accuracy': '{:.2f}'.format(train_acc),
        'validation_accuracy': '{:.2f}'.format(val_acc),
        'test_top1_accuracy': '{:.2f}'.format(top1acc),
        'test_top5_accuracy': '{:.2f}'.format(top5acc),
        'batch_size': batch_size,
        'epochs_specified': epochs,
        'epochs_trained': epoch,
        'what_was_not_trained': vgg_no_train_parts,
        'which_optimizer_used': "{} with a lr of {} and momentum {}".format(optimizer_str, learning_rate, momentum),
        'criterion_used': 'CrossEntropyLoss',
        'time_used_for_training': time_elapsed
    }
    save_to_json(results, results_path)


if __name__ == "__main__":
    main()


