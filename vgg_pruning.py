import torch
import torch.nn as nn
from datetime import datetime
import os
from os import path
import argparse
import pandas as pd

# my own functions
from helpers import set_device,\
    imagenette_dataloader,\
    create_optimizer,\
    train_model, \
    test_model,\
    load_model,\
    save_model
from pruning_methods import LocalMagnitudeUnstructured, LocalRandomUnstructured


def main():
    parser = argparse.ArgumentParser(description='Start skript for VGG16 Training runs')
    parser.add_argument('compression_rate', type=int, nargs='*', help='Specify the compression rate if not an experiment')

    arguments = parser.parse_args()
    compression_rate = arguments.compression_rate

    pruning_experiments(compression_rate)


def pruning_experiments(compression_rate):
    if not compression_rate:
        sparsities = [2, 4, 8, 16, 32, 64]  # compression ratios as defined in blalock
    else:
        sparsities = compression_rate
    print('Pruning with following sparsities: {}'.format(sparsities))

    if not sparsities:
        raise Exception('No Sparsities specified!')

    # results dictionary, later for saving results
    results = {
        'pruning_method': [],
        'sparsity': [],
        'modules_pruned_and_finetuned': [],
        'pruned_model_path': [],
        'batch_size': [],
        'max_epochs_finetuned': [],
        'early_stopping_epoch': [],
        'training_accuracy': [],
        'validation_accuracy': [],
        'time_elapsed': [],
        'test_top1_accuracy': [],
        'test_top5_accuracy': []
    }

    pruning_methods = [
        'local_magnitude_unstructured'
        # , 'local_random_unstructured'
    ]

    for pruning_method in pruning_methods:
        # specify pre-trained model
        device = set_device()
        model = load_model('trained_imgnette_models/run8_pretrained_100ep_ft_no_normalization.pt', device)

        # load imagenette
        batch_size = 64
        trainloader, validloader, testloader = imagenette_dataloader(batch_size)

        for sparsity in sparsities:
            print('-' * 20)
            print('---METHOD: {}, SPARSITY: {}---'.format(pruning_method, sparsity))
            pruning_percentage = 1-1 / sparsity   # PyTorch does not prune already pruned parameters - therefore the percentage stays at 0.5 when compression doubles.

            # PRUNING
            if pruning_method == 'local_magnitude_unstructured':
                local_magnitude_unstructured = LocalMagnitudeUnstructured()
                pruning_method = local_magnitude_unstructured(model)  # standard pruning_percentage 0.5
            elif pruning_method == 'local_random_unstructured':
                local_random_unstructured = LocalRandomUnstructured()
                pruning_method = local_random_unstructured(model)
            else:
                # Implement additional pruning methods
                raise NotImplementedError()

            print('-' * 10)
            print('--- Acc before Finetuning:---')
            test_model(model, testloader, device)

            optimizer = create_optimizer(list(model.parameters()), 'SGD', 0.001, 0.9)   # same settings as in VGG Training so far
            epochs = 50
            criterion = nn.CrossEntropyLoss()
            best_model, es_epoch, train_acc, val_acc, val_acc_history, time_elapsed = train_model(model, trainloader, validloader, epochs, optimizer, criterion, device)

            # TESTING
            top1_accuracy, top5_accuracy = test_model(model, testloader, device)

            # Save Model
            pruned_model_folder = 'pruned_models/{}-pretrained'.format(pruning_method)
            if not path.exists(pruned_model_folder):
                os.mkdir(pruned_model_folder)

            # A model is saved after every pruning + fintuning iteration
            now = datetime.now()
            model_path = save_model(model, pruned_model_folder + '/{}-compression_{}.pt'.format(sparsity, now.strftime("%d-%m-%Y_%H-%M-%S")))


            # SAVE RESULTS ONCE FOR ALL PRUNING METHODS - ALL RESULTS IN ONE FILE
            # Append Results - CSV Format
            results['pruning_method'].append(pruning_method)
            results['sparsity'].append(sparsity)
            results['modules_pruned_and_finetuned'].append('features')
            results['pruned_model_path'].append(model_path)
            results['batch_size'].append(batch_size)
            results['max_epochs_finetuned'].append(epochs)
            results['early_stopping_epoch'].append(es_epoch)
            results['training_accuracy'].append(train_acc)
            results['validation_accuracy'].append(val_acc)
            results['time_elapsed'].append(time_elapsed)
            results['test_top1_accuracy'].append(top1_accuracy)
            results['test_top5_accuracy'].append(top5_accuracy)

    results_df = pd.DataFrame.from_dict(results)
    now = datetime.now()
    results_path = 'pruning_results/results_{}.csv'.format(now.strftime("%d-%m-%Y_%H-%M-%S"))
    results_df.to_csv(results_path)


if __name__ == "__main__":
    main()


