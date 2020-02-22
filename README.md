# SIIM-ACR Pneumothorax Segmentation

Repository for the code of siim-acr-pneumothorax-segmentation challenge.

-   To train the model just run the start_training.py script. There you provide the path to your dataset and other training parameters. 
Furthermore you can also specify whether you want to train the full densenet or the tiramisu encoder. It is also possible to specify the type of loss function and the type of weighting in the batches
-   The heatmaps can be generated from the heatmaps.py script
-   The notebook model-for-colab.ipynb is suitable for training in Google Colab. For running it, the dataset needs to be extracted in the '/content' directory in Google Colab. If you want to extract it somewhere else, just change the path.
-   In the dataset resizing notebook is the function used for resizing the dataset to (224, 224).
    It heavily reduces the dataset size and is only done once, instead of doing it repetitively in the data loader.



#
-   Part of the code is based on the CheXNet replication from jrzech.
-   The model for tiramisu is based on the implementation from baldassarreFe