from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import csv
import cxr_dataset as CXR
import auc_calculation
import model_tiramisu


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#feature_extracting = True, means that we only want to traing the last layer(classification layer) and not finetune the entire network
def setup_and_train(DATA_DIR, BATCH_SIZE, LR, WEIGHT_DECAY, num_epochs, model_type, initial_model_path=None, feature_extracting = False, stop_if_no_improvement = True, decay_lr = True, print_time_multiplier = 1000, loss_function = 'BCELossLogits2'):
    num_classes = 14
    model_ft = initialize_model(num_classes, feature_extracting, model_type, use_pretrained_weights=True)
    print('Device: {0}'.format(device))
    # print(model_ft)
    dataloaders_dict, dataset_sizes = load_data(DATA_DIR, BATCH_SIZE)

    #send the model to GPU if available
    model_ft = model_ft.to(device)

    optimizer_ft = create_optimizer(model_ft, LR, WEIGHT_DECAY)
    print(optimizer_ft)

    criterion = nn.BCEWithLogitsLoss()

    model_ft, best_epoch, best_epoch_train, stats, LR = train_model(model_ft, criterion, optimizer_ft, LR, num_epochs,
                                                                    dataloaders_dict, dataset_sizes, WEIGHT_DECAY,
                                                                    decay_lr=False, print_time_multiplier=1, loss_function = loss_function)

    visualize_loss(stats)

    preds, aucs, trues = auc_calculation.make_pred_multilabel(model_ft, DATA_DIR)
    print(aucs)


def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        WEIGHT_DECAY, stop=True, decay_lr=True, print_time_multiplier=1000, loss_function = 'BCELossLogits'):

    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELossWithLogits)
        optimizer: optimizer to use in training (SGD or Adam)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    print(model)
    print('Num of epochs:', num_epochs)
    since = time.time()
    stats = {
        'val_loss_history': [],
        'train_loss_history': []
        }
    #     val_loss_history = []
    #     train_loss_history = []
    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    best_loss_train = 999999
    best_epoch_training = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            # running_loss2 = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            for idx, data in enumerate(dataloaders[phase]):
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.float().to(device))
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                #Use this in case we want to weight both, positive and negative samples
                if loss_function == "BCELossLogits":
                    beta = pos_neg_weights_in_batch(labels)
                    print(beta)
                    criterion = nn.BCEWithLogitsLoss(pos_weight=beta)
                    loss = criterion(outputs, labels)
                # Use this in case we want only want to weight positive samples
                else:
                    P = 0
                    N = 0
                    for idxi, label in enumerate(labels):
                        for v in label:
                            if int(v) == 1:
                                P = P + 1
                            else:
                                N = N + 1
                    if P!=0 and N!=0:
                        BP = (P + N)/P
                        BN = (P + N)/N
                        weights = torch.tensor([BP, BN], dtype=torch.float).to(device)
                        print(BP, BN)
                    else: weights = None

                    loss = weighted_BCELoss(outputs, labels, weights=weights)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * batch_size

                if ((idx + 1) % print_time_multiplier == 0):
                    print("Loss of iteration {} is: {}".format(idx + 1, loss.item() * batch_size))

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                last_train_loss = epoch_loss
                stats['train_loss_history'].append(epoch_loss)
            if phase == 'val':
                stats['val_loss_history'].append(epoch_loss)

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            if phase == 'val' and epoch_loss > best_loss and decay_lr:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                print('The new learning rate is: ', LR)
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=WEIGHT_DECAY)
                # optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, betas=(0.9, 0.999))
                print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'train' and epoch_loss < best_loss_train:
                best_loss_train = epoch_loss
                best_epoch_train = epoch
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR, "results")

            #             log training and validation loss over each epoch
            if phase == 'val':
                with open("results/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if (epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if (total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3) and stop:
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    #     load best model weights to return
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch, best_epoch_train, stats, LR

def weighted_BCELoss(output, target, weights=None):
    output = output.clamp(min=1e-5, max=1-1e-5)
    target = target.float()
    if weights is not None:
        assert len(weights) == 2

        loss = -weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
    else:
        loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)

    return torch.sum(loss)

def pos_neg_weights_in_batch(labels_batch):
    num_total = labels_batch.shape[0] * labels_batch.shape[1]
    num_positives = labels_batch.sum()
    num_negatives = num_total - num_positives

    if not num_positives == 0:
        beta_p = num_negatives / num_positives
    else:
        beta_p = num_negatives
    # beta_p = torch.tensor(beta_p)
    beta_p = beta_p.to(device)
    beta_p = beta_p.type(torch.cuda.FloatTensor)

    return beta_p

def checkpoint(model, best_loss, epoch, LR, path):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """
    # os.chdir('/content/drive/My Drive/siim-acr-pneumothorax-segmentation-drive')
    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, path + "/checkpoint")


def initialize_model(num_classes, feature_extracting, model_type, use_pretrained_weights=True, loss_function = 'BCELossLogits'):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if model_type == 'full':
        model_ft = models.densenet121(pretrained=use_pretrained_weights)

        # If feature_extracting true, only the classifier which is defined below will be finetuned.
            # This will set required_grad attribute for all the other parameters to false so we do not have to compute gradients
        if feature_extracting:
            for param in model_ft.parameters():
                param.requires_grad = False

        num_ftrs = model_ft.classifier.in_features

        # Use this in case you use BCEWithLogitsLoss, where the sigmoid is applied internally by the loss function -> More numerically stable
        if loss_function == "BCELossLogits":
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        else:
            # Use this in case you use BCELoss, where the sigmoid is applied by the network itself
            model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
    #enter here for the tiramisu encoder
    else:
        model_ft = model_tiramisu.FCDenseNet103(in_channels=3,
                                 out_channels=14,
                                 dropout=0.2)

    return model_ft

def load_data(DATA_DIR, BATCH_SIZE):
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # the commented transformations can be skipped when using the resized dataset
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]),
        'val': transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]),
        }

    # Create training and validation datasets
    image_datasets = {}
    image_datasets['train'] = CXR.CXRDataset(
        path_to_images=DATA_DIR,
        fold='train',
        sample = 1,
        transform=data_transforms['train'])
    image_datasets['val'] = CXR.CXRDataset(
        path_to_images=DATA_DIR,
        fold='val',
        sample = 1,
        transform=data_transforms['val'])

    # Create training and validation dataloaders
    dataloaders_dict = {}
    dataloaders_dict['train'] = torch.utils.data.DataLoader(
        image_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders_dict['val'] = torch.utils.data.DataLoader(
        image_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders_dict, dataset_sizes

def create_optimizer(model, LR, WEIGHT_DECAY, type = 'SGD'):
    if type == "SGD":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, betas=(0.9, 0.999))

    return optimizer

def visualize_loss(stats):
    plt.subplot(2, 1, 2)
    plt.plot(stats['train_loss_history'], label='train')
    plt.plot(stats['val_loss_history'], label='val')
    plt.title('Train and validation loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def count_num_parameters(model):
    num_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            num_parameters = num_parameters + 1
    return num_parameters





