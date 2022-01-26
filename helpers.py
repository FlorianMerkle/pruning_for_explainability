import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision.datasets as datasets

import time
import json
import copy


def set_device():

    """
    Function to identify if CUDA device available and if so, sets device to CUDA. Otherwise CPU is chosen
    :return: device, used to move data and model to this device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
        print('--- cuda device set ---')
    else:
        device = 'cpu'
        print('--- stays on cpu - no cuda device available---')

    return device


def load_run_config(config_path, desired_run):
    with open(config_path) as file:
        config = json.load(file)
        try:
            if config[desired_run]:
                return config[desired_run]
        except KeyError as ke:
            raise KeyError('{} could not be found in {}'.format(ke, config_path))


def imagenette_dataloader(batch_size):
    train_transform = transforms.Compose([
        # Data Augmentation:
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),

        transforms.Resize((224, 224)),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Imagenet Values
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Imagenet Values
    ])

    trainset = datasets.ImageFolder('data/imagenette2/train', transform=train_transform)
    dataset = datasets.ImageFolder('data/imagenette2/val', transform=test_transform)
    testset, validset = torch.utils.data.random_split(dataset, [2800, 1125], generator=torch.Generator().manual_seed(69))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainloader, validloader, testloader


def disable_params_grad(model, which_parts):

    """
    Function to disable the gradient of the model parameters, as needed for feature extraction transfer learning.
    :param model: model from which the parameters' gradients should be removed
    :param which_parts: the parts of the model where grads will be disabled
    """
    if which_parts:
        print('-' * 10)
        for part in which_parts:
            print('Which part of the nn:', part)
            for layer in getattr(model, part):
                for name, param in layer.named_parameters():
                    param.requires_grad = False
                    print('\t Deactivated gradients for: {}, {}'.format(layer, name))


def enable_params_grad(model, which_parts):
    """
    Function to enable the gradient of the model parameters.
    :param model: model from which the parameters' gradients should be added
    :param which_parts: the parts of the model where grads will be enabled
    """
    if which_parts:
        for part in which_parts:
            print('Which part of the nn:', part)
            for layer in getattr(model, part):
                for name, param in layer.named_parameters():
                    param.requires_grad = True
                    print('\t Activated gradients for: {}, {}'.format(layer, name))


def get_params_grad(model):
    params_w_grad = []
    print('Params with gradient that are used for learning:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_w_grad.append(param)
            print('\t', name)

    return params_w_grad


def create_optimizer(params_w_grad, optimizer, learning_rate, momentum):

    optim_dict = {
        'SGD': optim.SGD(params_w_grad, lr=learning_rate, momentum=momentum, weight_decay=0.0001),
        'Adam': optim.Adam(params_w_grad, lr=learning_rate)
        # implement more
    }

    try:
        r_optim = optim_dict[optimizer]
    except KeyError as ke:
        raise KeyError('Desired Optimizer {} is not implemented yet!'.format(optimizer))

    return r_optim


def train_model(model, trainloader, validloader, num_epochs, optimizer, criterion, device):
    starting_time = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_acc = 0.0
    best_val_acc = 0.0

    # Early Stopping
    improving_delta = 1.01
    epochs_no_improv = 0
    epochs_early_stopping = 10

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch: {}/{}'.format(epoch + 1, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # set model to training mode
                dataloader = trainloader
            else:
                model.eval()    # set model to eval mode
                dataloader = validloader

            running_loss = 0
            running_correct = 0

            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                # zero out gradients
                optimizer.zero_grad()

                # forward
                # track when in training
                with torch.set_grad_enabled(phase == 'train'):
                    outs = model(images)
                    loss = criterion(outs, labels)

                    _, preds = torch.max(outs.data, 1)

                    # backward & optimize when in training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_correct += (preds == labels).sum().item()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_correct / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train' and epoch_acc > best_train_acc:
                best_train_acc = epoch_acc
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_val_acc * improving_delta:
                    print('INFO: Epoch Acc {} better than current Val Acc {} by {}'.format(epoch_acc, best_val_acc, epoch_acc-best_val_acc))
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improv = 0
                    es_epoch = epoch
                else:
                    epochs_no_improv += 1
                    print('INFO: Early Stopping Counter: {}'.format(epochs_no_improv))

            # early stopping, check every epoch
        if epochs_no_improv == epochs_early_stopping:
            print('Model has not improved in last {} epochs, early stopping in {} epoch!'.format(epochs_early_stopping, epoch + 1))
            break

    time_elapsed = time.time() - starting_time
    time_elapsed = "{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
    print('Training completed in {}'.format(time_elapsed))

    model.load_state_dict(best_model_wts)

    return model, es_epoch, best_train_acc, best_val_acc, val_acc_history, time_elapsed


def test_model(model, testloader, device):
    top1 = 0
    top5 = 0

    model.eval()
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        top1 += topk_correct(preds, labels, (1, 5))[0]
        top5 += topk_correct(preds, labels, (1, 5))[1]

    top1_accuracy = top1 / len(testloader.dataset)
    top5_accuracy = top5 / len(testloader.dataset)

    print('Top-1 Accuracy on Test-Set: {} \n Top-5 Accuracy on Test-Set: {}'.format(top1_accuracy.item(), top5_accuracy.item()))
    return top1_accuracy.item(), top5_accuracy.item()


def save_model(model, path):
    if not path.endswith('.pt'):
        file_ext = '.pt'
        path = path + file_ext

    torch.save(model, path)
    print('-' * 10)
    print('Saved Model at: {}'.format(path))
    return path


def save_to_json(results, path):
    with open(path, 'a') as file:
        json.dump(results, file)
    print('Results saved to {}'.format(path))


def load_model(model_path, device):
    map_location = torch.device(device)

    if model_path.endswith('.pt'):
        model = torch.load(model_path, map_location=map_location)
    else:
        file_ext = '.pt'
        model = torch.load(model_path + file_ext, map_location=map_location)

    return model


def topk_correct(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk correct [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (
                    y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(
                -1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0,
                                                                                        keepdim=True)  # [kB] -> [1]
            list_topk_accs.append(tot_correct_topk)
        return list_topk_accs

