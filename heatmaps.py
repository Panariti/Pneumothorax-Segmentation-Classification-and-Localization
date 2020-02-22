from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import cv2
import cmapy
import cxr_dataset as CXR
from PIL import Image
class DenseNet(nn.Module):
    def __init__(self, model_ft):
        super(DenseNet, self).__init__()

        # get the pre-trained DenseNet121 network
        self.densenet = model_ft

        # disect the network to access its last convolutional layer
        self.features_conv = self.densenet.features
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = self.densenet.classifier
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        x = torch.nn.functional.relu(x, inplace=True)
        # register the hook
        h = x.register_hook(self.activations_hook)
        x = self.global_avg_pool(x)
        x = x.view((1, 1024))
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)

def load_data(label='any'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    PATH_TO_IMAGES = "dataset/images"
    finding = 'any'
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    labels = [
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
        'Hernia']
    if label in labels:
        finding = label

    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='test',
        transform=data_transform, finding=finding)
    print('Total number of images is: {0}'.format(len(dataset.df)))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    return iter(dataloader), len(dataset.df)


def load_model(path_to_model):
    checkpoint = torch.load(path_to_model, map_location=lambda storage, loc: storage)
    model_ft = checkpoint['model']
    densenet = DenseNet(model_ft)
    densenet.eval()

    return densenet



def generate_cams(data_dir, path_to_model, save_here, label = '', how_many_to_show = 'all'):
    # pneumothorax_index = 8
    dataloader, num_samples = load_data(label='Pneumothorax')
    if how_many_to_show == 'all':
        how_many_to_show = num_samples

    labels = ['Atelectasis',
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
              'Hernia']
    label_index = labels.index(label)
    # print('label_index', label_index)
    densenet = load_model(path_to_model)
    for i in range(how_many_to_show):
        img, truth, image_name = next(dataloader)

        # name of the image in the dataset
        name = image_name[0]
        name_without_type = image_name[0][0:len(image_name[0]) - 4]
        pred = densenet(img)
        pred[:, label_index].backward()
        gradients = densenet.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = densenet.get_activations(img).detach()
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        image = cv2.imread(data_dir + '/{0}'.format(name))
        heatmap_real = heatmap
        heatmap = cv2.resize(np.float32(heatmap), (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cmapy.cmap('viridis'))
        superimposed_img = heatmap * 0.4 + image

        image_location = save_here + '/heatmap-{0}.jpg'.format(name_without_type)
        cv2.imwrite(image_location, superimposed_img)
        heatmap_image = Image.open(image_location)

        f, axarr = plt.subplots(1, 2)
        axarr[0].matshow(heatmap_real.squeeze())
        axarr[1].imshow(heatmap_image)
        plt.savefig(save_here + '/heatmap-image-{0}.jpg'.format(name_without_type))
        # plt.imshow(heatmap_image)
        plt.show()


if __name__ == '__main__':
    generate_cams('data_dir', 'path_to_model', 'heatmaps', label = 'Pneumothorax', how_many_to_show=5)
