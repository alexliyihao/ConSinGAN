from cifar10_web import cifar10
import numpy as np
import torch
import os
import skimage.io as io
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt

def pipeline_rep_selection(imbalanced_label: int = 0,
                           drop_ratio: float = 0.4,
                           num_representatives: int = 40,
                           PCA__n_component: int = 50,
                           DBSCAN__eps: float = 0.75,
                           DBSCAN__min_sample: int = 3,
                           parent_directory= None):

    if parent_directory == None:
        parent_directory = os.getcwd()

    save_directory = generate_save_directory(imbalanced_label,
                                             drop_ratio,
                                             num_representatives,
                                             PCA__n_component,
                                             DBSCAN__eps,
                                             DBSCAN__min_sample)

    save_directory = os.path.join(parent_directory, save_directory)

    if os.path.exists(save_directory):
        print("the representatives are already generated")
        return 0

    else:
        os.makedirs(save_directory)
        os.makedirs(os.path.join(save_directory, "rep_selected"))

        X_train, y_train, X_test, y_test = get_cifar_10()
        X_train_imbalanced, y_train_imbalanced, X_deleted, y_deleted= get_imbalanced_dataset(X_train, y_train, label = imbalanced_label, drop_ratio= drop_ratio)
        imbalanced_class = X_train_imbalanced[y_train_imbalanced == imbalanced_label]

        vgg19_extracted_feature = vgg19_lize(imbalanced_class)
        PCA_extracted_feature = run_PCA(vgg19_extracted_feature, PCA__n_component)
        DBSCAN_label = run_DBSCAN(PCA_extracted_feature, eps = DBSCAN__eps, min_sample = DBSCAN__min_sample)
        rep_distribution = get_rep_distribution(N_represenative = num_representatives, label = DBSCAN_label)
        cluster_p1 = get_sub_cluster_p1(data = PCA_extracted_feature, label = DBSCAN_label)

        rep_index_list = []
        #for each cluster(usually only one)
        for i in range(len(cluster_p1)):

            sub_cluster = cluster_p1[i]
            num_rep = rep_distribution[i]

            #run a GMM to find the mean
            GMM = GaussianMixture(n_components = num_rep)
            GMM.fit(sub_cluster)
            means = GMM.means_

            #find the closest image
            reps = find_closest(cluster = sub_cluster, means = means)
            #add them to the closest image
            rep_index_list += list(np.where(PCA_extracted_feature == rep)[0][1] for rep in reps)

        rep_list = imbalanced_class[np.random.choice(np.array(rep_index_list), size = num_representatives, replace = False)]

        for i, images in enumerate(rep_list):
            images = (images*255).astype(np.uint8)
            io.imsave(os.path.join(save_directory, f"rep_selected/rep_{i}.png"), images)
        return rep_list, save_directory

def get_indexes(index_list, label: int = 5, drop_ratio: float = 0.4):

    drop_ratio_list = dict(zip([0.4, 0.6, 0.75, 0.9],range(4)))

    return index_list[label+10*drop_ratio_list[drop_ratio]]

def get_cifar_10(return_one_hot_y: bool = False):

    X_train, y_train, X_test, y_test = cifar10(path=None)

    X_train = X_train.reshape(-1,3,32,32).transpose(0,2,3,1)
    X_test = X_test.reshape(-1,3,32,32).transpose(0,2,3,1)

    if return_one_hot_y == False:
        y_train = np.array([np.argmax(a, axis=0) for a in y_train])
        y_test = np.array([np.argmax(a, axis=0) for a in y_test])

    return X_train, y_train, X_test, y_test

def get_imbalanced_dataset(X_train, y_train, label, drop_ratio):

    if isinstance(label,int) and isinstance(drop_ratio, float):
        label = [label]
        drop_ratio = [drop_ratio]
    else:
        label = list(label)
        drop_ratio = list(drop_ratio)

    assert(len(label) == len(drop_ratio))
    assert(sum([1 if i in [0.4, 0.6, 0.75, 0.9] else 0 for i in drop_ratio]) == len(drop_ratio))

    npzfile = np.load("selected_index_40.npz", allow_pickle = True)
    indexes = npzfile["arr_0"]

    if y_train.ndim == 2:
        y_train_ = np.array([np.argmax(a, axis=0) for a in y_train])
    else:
        y_train_ = y_train

    for label_, drop_ratio_, i in zip(label, drop_ratio, range(len(label))):

        if i == 0:
            label_index = np.where(y_train_ == label_)[0]
            sample_index = get_indexes(indexes, label = label_, drop_ratio = drop_ratio_)
            deleted_index = np.delete(label_index, sample_index)
        else:
            label_index = np.where(y_train_ == label_)[0]
            sample_index = get_indexes(indexes, label = label_, drop_ratio = drop_ratio_)
            print(deleted_index.shape)
            print(np.delete(label_index, sample_index).shape)
            deleted_index = np.concatenate((deleted_index, np.delete(label_index, sample_index)))


    X_imbalanced = np.delete(X_train, deleted_index, 0)
    y_imbalanced = np.delete(y_train, deleted_index, 0)

    X_deleted = X_train[deleted_index]
    y_deleted = y_train[deleted_index]

    return X_imbalanced, y_imbalanced, X_deleted, y_deleted

def vgg19_lize(data):
    toTensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    normalized = transforms.Compose([toTensor, normalize])
    vgg19 = models.vgg19(pretrained=True)
    vgg19 = vgg19.cuda()
    result = torch.stack([normalized(x) for x in data]).cuda()
    features = []
    vgg19.eval()
    batch_size = 30
    for i in range(0, result.shape[0], batch_size):
        batch = result[i:(i+batch_size)]
        with torch.no_grad():
            features.append(vgg19.features(batch).data.cpu().detach().numpy())
    features = np.vstack(features)
    features = features.reshape(features.shape[0], -1)
    return features

def run_PCA(features, n_components: int = 50):
    features = (features - features.min())/(features.max() - features.min())
    pca = PCA(n_components=n_components)
    extracted_features = pca.fit_transform(features)
    return extracted_features

def run_DBSCAN(features, eps: float = 0.75, min_sample: int = 3):
    dbscan = DBSCAN(eps = eps, min_samples = min_sample)
    dbscan.fit(features)
    cluster_label = dbscan.labels_
    return cluster_label

def get_rep_distribution(N_represenative, label):
    """
    given the cluster label generated by DBSCAN, calculate the number of representative for each cluster

    input:
        label: numpy.ndarray, the output of sklearn.cluster.DBSCAN's label_ method
    output:
        rep_distribution: numpy.ndarray, the number of representative for each cluster, in the same order
    """
    label_, count = np.unique(label, return_counts = True)
    #remove the noise term(cluster = -1)
    count = count[1:]
    rep_distribution = np.ceil(count*N_represenative/count.sum()).astype("int")
    return rep_distribution

def get_sub_cluster_p1(data, label):
    """
    given the original dataset and the label from sklearn.cluster, return the list of each cluster

    input:
        data: numpy.ndarray, original data passed into sklearn.cluster.DBSCAN's fit method
        label: numpy.ndarray, the output of sklearn.cluster.DBSCAN's label_ method
    output:
        cluster_p1: list of numpy.ndarray, the feature divided by labels
    """
    unique_label = np.unique(label)[1:]
    cluster_p1 = [[] for i in np.arange(unique_label.shape[0])]
    for i in np.arange(np.shape(label)[0]):
        if (label[i] != -1):
            cluster_p1[label[i]].append(data[i])
    cluster_p1 = [np.array(i) for i in cluster_p1]
    return cluster_p1

def find_closest(cluster, means):
    """
    find the single sample(image) in the cluster which has smallest euclidean distance to mean

    input:
        cluster: numpy.ndarray, a set of sample(image)
        mean: numpy.ndarray, an array of means, from the output of sklearn.mixture.GMM.means_

    output:
        rep: numpy.ndarray, the sample in the cluster which has smallest euclidean distance to each mean respectively
    """

    rep_index = pairwise_distances_argmin(means,cluster)
    return cluster[rep_index]

def generate_save_directory(imbalanced_label, drop_ratio, num_representatives, PCA__n_component, DBSCAN__eps, DBSCAN__min_sample):
    return f"label_{str(imbalanced_label)}_drop_ratio_{drop_ratio}_num_representatives_{num_representatives}_PCA__n_component_{PCA__n_component}_DBSCAN__eps_{DBSCAN__eps}_DBSCAN__min_sample_{DBSCAN__min_sample}"
