import torch
import matplotlib.pyplot as plt

from mnist_dataset import MnistDataset

mnist_dataset_train = MnistDataset('mnist_data/mnist_train.csv')

from gan_mnist_generator import Generator
from gan_mnist_discriminator import Discriminator

from gan_mnist_utils import generate_random_seed, label_true, label_false

D = Discriminator()
G = Generator()

# training
epochs = 3

for i in range(epochs):
    print('training epoch', i, 'of', epochs)
    for label, image_data_tensor, target_tensor in mnist_dataset_train:    
        D.train(image_data_tensor, label_true)
        D.train(G.forward(generate_random_seed(100)).detach(), label_false)
        G.train(D, generate_random_seed(100), label_true)
        pass
    pass

D.save_model()
G.save_model()

# display the progress
D.plot_progress()
G.plot_progress()

plt.show()
