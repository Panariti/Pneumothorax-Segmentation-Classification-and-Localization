import torch
import pandas as pd
import cxr_dataset as CXR
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_pred_multilabel(model, PATH_TO_IMAGES, BATCH_SIZE = 4):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: the model to calculate the AUCs for
        PATH_TO_IMAGES: path at which NIH images can be found
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    # calc preds in batches of 16, can reduce if your GPU has less RAM

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(224),
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
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

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    # create dataloader
    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold="test",
        sample = 200,
        transform=data_transforms['val'])
    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)
    size = len(dataset)
    print(size)
    # print('datasetsize', dataset_sizes)
    print(dataset.df)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # these lists will save values for the second way of calculating the scores
    outputList = []
    labelList = []

    # iterate over dataloader
    for idx, data in enumerate(dataloader):

        inputs, labels, _ = data
        inputs = Variable(inputs.to(device))
        labels = Variable(labels.float().to(device))

        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        probs = outputs.cpu().data.numpy()

        for i in range(outputs.shape[0]):
            outputList.append(outputs[i].tolist())
            labelList.append(labels[i].tolist())

        # get predictions and true values for each item in batch
        # here the dataframe 'true' and 'preds' are created by adding rows into them respectively
        for j in range(0, batch_size[0]):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * idx + j]
            truerow["Image Index"] = dataset.df.index[BATCH_SIZE * idx + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            for k in range(len(dataset.PRED_LABEL)):
                thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]
            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)
            # print(pred_df)
            # print(head(true_df))
        if (idx % 100 == 0):
            print(str(idx * BATCH_SIZE))


    #Another way to calculate the AUCs. The calculation is done step by step.
    print('Scores - Method2 -----------------------')
    epoch_auc_ave = sklm.roc_auc_score(np.array(labelList), np.array(outputList))
    epoch_auc = sklm.roc_auc_score(np.array(labelList), np.array(outputList), average=None)
    for i, c in enumerate(dataset.PRED_LABEL):
        fpr, tpr, _ = sklm.roc_curve(np.array(labelList)[:, i], np.array(outputList)[:, i])
        plt.plot(fpr, tpr, label=c)
        print('{}: {:.4f} '.format(c, epoch_auc[i]))
    print('Scores - Method2 -----------------------')

    # here the auc scores are calculated and the 'auc' table is created
    auc_df = pd.DataFrame(columns=["label", "auc"])

    # calc AUCs
    for column in true_df:

        if column not in [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']:
            continue
        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:
            thisrow['auc'] = sklm.roc_auc_score(
                actual.values.astype(int), pred.values)
        except BaseException as e:
            print('-------------------')
            print(e)
            print(actual.values)
            print(pred.values)
            print('-------------------')

        auc_df = auc_df.append(thisrow, ignore_index=True)

    pred_df.to_csv("results/preds.csv", index=False)
    auc_df.to_csv("results/aucs.csv", index=False)
    true_df.to_csv('results/true.csv', index = False)
    return pred_df, auc_df, true_df