import time
import torch
import torch.utils.data
import torch.optim as optim
from loss import *
from torch.autograd import Variable
from model import model
from Evaluation import evaluation
from torchvision import transforms
from sklearn.datasets import fetch_lfw_people
from utils import GenIdx,IdentitySampler, prepare_dataset, prepare_set, \
                    prepare_data_ids,LFW_training_Data
from loss import contrastive_loss, total_loss,tf
import numpy as np
from tensorflow import keras
from keras.optimizers import Adam

# Test de prepare_dataset

# Work on GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import the lfw dataset - With 15 img min per person
# print("===> Loading Fetch Dataset \n")
lfw_people = fetch_lfw_people(min_faces_per_person=15, resize=0.4)
nb_img_per_id_to_keep = 15

# Renvoie la base de données avec 15 images par personnes maximum
# Vous devrez coder la fonction dans le fichier utils
X, y, n_classes = prepare_dataset(lfw_people, nb_img_per_id_to_keep)
# print("Nombres de classes : ",n_classes)
# print("Nombre d'images ", len(X))

# Produire les listes d'identités qui correspondront aux folds d'entraînement / validation et
# aux identités de test.
# train_ids_lists et val_ids_lists seront des listes de 5 listes d'identités (Une liste pour chaque fold)
# test_ids sera une liste de 21 identités
train_ids_lists, val_ids_lists, test_ids = prepare_data_ids(n_classes)
# print("train_ids : ",train_ids_lists)
# print("val_ids : ",val_ids_lists)
# print("test_ids : ",test_ids)

 # Prepare var - Batch size - On utilise 4 images par identités et 8 identités par batch (=> 32 imgs)
num_img_of_same_id_in_batch = 4
num_different_identities_in_batch = 8
batch_size = num_img_of_same_id_in_batch * num_different_identities_in_batch # batch_size = 32
test_batch_size = 32

global_img_pos = GenIdx(y)  # Get the images positions in list for each specific identity


folds = 5  # Number of fold K for cross-validation
epochs = 10 # Number of epochs

# Augmentation de la taille des images
img_h, img_w = 288, 144

# Préparation des transformations appliquées aux images pour l'entraînement ou le reste (validation / test)
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_h, img_w)),
    transforms.Pad(10),
    transforms.RandomCrop((img_h, img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
])




