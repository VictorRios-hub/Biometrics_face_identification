# Allow to get the complement of a list slice
import numpy as np
from torch.utils.data.sampler import Sampler
import torch.utils.data as data
import random
import torch

# Renvoie la base de données d'images sous X et les labels sous y
# Conserve uniquement 15 images par personnes maximum
def prepare_dataset(lfw_people, nb_img_per_id_to_keep):
    n_samples, h, w = lfw_people.images.shape

    X_init = lfw_people.images
    y_init = lfw_people.target

    X = []  
    y = []  

    # Dictionary pour comptage
    person_image_count = {}

    # Pour chaque imgaes
    for i in range(n_samples):
        image = X_init[i]
        label = y_init[i]

        if label in person_image_count:
            # Check moins de 15 
            if person_image_count[label] < nb_img_per_id_to_keep:
                X.append(image)
                y.append(label)
                person_image_count[label] += 1
        else:
            # new person
            X.append(image)
            y.append(label)
            person_image_count[label] = 1

    X = np.array(X)
    y = np.array(y)

    # Calculate the number of unique classes (persons)
    n_classes = len(np.unique(y))

    return X, y, n_classes



def prepare_data_ids(n_classes):
    # ids generate
    all_ids = list(range(n_classes))

    # test fold
    test_ids = all_ids[:21]

    # On réserve les 21 premiers labels pour test
    train_val_ids = all_ids[21:]

    # 5 ids par fold d'entraînement/validation
    fold_size = (n_classes -21) // 5

    # k_folds divise
    train_val_ids_lists = [train_val_ids[i:i + fold_size] for i in range(0, n_classes, fold_size)]

    # Séparer les 5 folds en 4 folds d'entraînement et 1 fold de validation à tour de rôle
    train_ids_lists = []
    val_ids_lists = []
    for i in range(5):
        val_ids_lists.append(train_val_ids_lists[i])
        train_ids_lists.append([id for ids in train_val_ids_lists[:i] + train_val_ids_lists[i + 1:] for id in ids])

    return train_ids_lists, val_ids_lists, test_ids

def extract_fold_subset(X, img_pos, img_ids_to_extract):
    # Initialize les listes 
    X_extracted = []
    y_extracted = []

    for idx, new_label in enumerate(img_ids_to_extract):

        for i in range (15):
            # Recup l'image et le label
            image = X[img_pos[img_ids_to_extract[idx]][i]]
            X_extracted.append(image)
            y_extracted.append(idx)

    return X_extracted, y_extracted


class prepare_set(data.Dataset):
    """

    Classe permettant la création d'un set pour validation ou tests.
    Vous aurez besoin de coder la fonction extract_fold_subset si ce n'est pas déjà fait.

    """
    def __init__(self, X, img_pos, img_ids_to_extract, transform=None):

        # Récupération du subset correspondant à un fold donné et relabel
        # La fonction extract_fold_data est à définir
        extracted_X, extracted_y = extract_fold_subset(X, img_pos, img_ids_to_extract)

        self.val_test_image = extracted_X
        self.val_test_label = extracted_y
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.val_test_image[index], self.val_test_label[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        #Should be the same len for both image 1 and image 2
        return len(self.val_test_image)

class LFW_training_Data(data.Dataset):
    def __init__(self, X, img_pos, img_ids_to_extract, transform=None, colorIndex=None):

        # Récupération du subset correspondant à un fold donné et relabel
        # La fonction extract_fold_data est à définir
        extracted_X, extracted_y = extract_fold_subset(X, img_pos, img_ids_to_extract)

        self.train_images = extracted_X
        self.train_labels = extracted_y

        self.transform = transform

        # Prepare index
        self.cIndex = colorIndex

    def __getitem__(self, index):
        #Dataset[i] return image with its corresponding label
        img, target = self.train_images[self.cIndex[index]], self.train_labels[self.cIndex[index]]

        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_labels)

# Sampler - Will select the right amount of ids and images per ids for the upcoming dataloader
class IdentitySampler(Sampler):

    """Sample person identities evenly in each batch.
        Args:
            train_labels : labels of each modalities
            color_pos : positions of each identity
            num_of_same_id_in_batch : Number of images per identity in a batch
            batch_num_identities: Number of identity in a batch
    """

    def __init__(self, train_labels, color_pos, num_of_same_id_in_batch, batch_num_identities):
        uni_label = np.unique(train_labels)
        self.n_classes = len(uni_label)
        N = len(train_labels)  # nb_img_per_id = 15
        for j in range(int(N / (batch_num_identities * num_of_same_id_in_batch)) + 1):
            # We choose randomly "num_different_identities_in_batch" identities :
            batch_idx = np.random.choice(uni_label, batch_num_identities, replace=False)
            for i in range(batch_num_identities):  # For each ids in a batch
                # We choose randomly "num_img_of_same_id_in_batch" images :
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_of_same_id_in_batch)
                if j == 0 and i == 0:
                    index = sample_color
                else:
                    index = np.hstack((index, sample_color))
        self.index = index
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index)))

    def __len__(self):
        return self.N

# Pour une liste y de labels, GenIdx génère une liste L de listes qui contiennent chacunes les positions des images pour un label donné.
# Par exemple pour le label 4, on pourra accéder à la liste des positions des images associées à ce label via L[4]
def GenIdx(train_label):
    color_pos = []
    unique_label_color = np.unique(train_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)
    return color_pos