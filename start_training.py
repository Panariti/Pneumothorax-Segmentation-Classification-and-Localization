import model as M


# you will need to customize PATH_TO_IMAGES to where you have uncompressed
# NIH images
DATA_DIR = ""
BATCH_SIZE = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
num_epochs = 1

if __name__ == '__main__':
    M.setup_and_train(DATA_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, num_epochs, model_type = 'tiramisu')

