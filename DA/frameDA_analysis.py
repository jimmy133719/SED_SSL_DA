import pdb
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from statsmodels.stats.weightstats import ztest

def visualization(ADA_synth_feature, ADA_real_feature, No_ADA_synth_feature, No_ADA_real_feature, ADA_path=None, No_ADA_path=None, sample_num=1000):
    print("start visualization")

    ADA_tsne_path = os.path.join(ADA_path, 'tsne(frame).npy')
    No_ADA_tsne_path = os.path.join(No_ADA_path, 'tsne(frame).npy')

    if not os.path.exists(ADA_tsne_path):
        synth_sample_index = np.random.choice(ADA_synth_feature.shape[0], sample_num)
        real_sample_index = np.random.choice(ADA_real_feature.shape[0], sample_num)
        ADA_synth_feature_sample = ADA_synth_feature[synth_sample_index]
        ADA_real_feature_sample = ADA_real_feature[real_sample_index]
        ADA_feature = np.concatenate((ADA_synth_feature_sample, ADA_real_feature_sample), axis=0)
        ADA_feature = scaler.fit_transform(ADA_feature)
        ADA_feature = TSNE(n_components=2).fit_transform(ADA_feature)
        np.save(ADA_tsne_path, ADA_feature)

    if not os.path.exists(No_ADA_tsne_path):
        No_ADA_synth_feature_sample = No_ADA_synth_feature[synth_sample_index]
        No_ADA_real_feature_sample = No_ADA_real_feature[real_sample_index]
        No_ADA_feature = np.concatenate((No_ADA_synth_feature_sample, No_ADA_real_feature_sample), axis=0)
        No_ADA_feature = scaler.fit_transform(No_ADA_feature)
        No_ADA_feature = TSNE(n_components=2).fit_transform(No_ADA_feature)
        np.save(No_ADA_tsne_path, No_ADA_feature)
    
    ADA_feature = np.load(ADA_tsne_path)
    No_ADA_feature = np.load(No_ADA_tsne_path)

    ADA_synth_feature_sample = ADA_feature[:sample_num]
    ADA_real_feature_sample = ADA_feature[sample_num:]
    No_ADA_synth_feature_sample = No_ADA_feature[:sample_num]
    No_ADA_real_feature_sample = No_ADA_feature[sample_num:]

    plt.figure()
    plt.scatter(ADA_synth_feature_sample[:,0], ADA_synth_feature_sample[:,1], alpha=0.6, s=10, marker='x', label='synth')
    plt.scatter(ADA_real_feature_sample[:,0], ADA_real_feature_sample[:,1], alpha=0.3, s=10, c='r', label='real')
    plt.legend()
    plt.title('Adaptation')
    plt.xlim(-60, 80)
    plt.ylim(-90, 90)
    plt.axis('off')
    plt.savefig(os.path.join(ADA_path, 'Adaptation(frame).png'))

    plt.figure()
    plt.scatter(No_ADA_synth_feature_sample[:,0], No_ADA_synth_feature_sample[:,1], alpha=0.6, s=10, marker='x', label='synth')
    plt.scatter(No_ADA_real_feature_sample[:,0], No_ADA_real_feature_sample[:,1], alpha=0.3, s=10, c='r', label='real')
    plt.legend()
    plt.title('No_adaptation')
    plt.xlim(-60, 80)
    plt.ylim(-90, 90)
    plt.axis('off')
    plt.savefig(os.path.join(No_ADA_path, 'No_adaptation(frame).png'))

def svm_classfication(ADA_synth_feature, ADA_real_feature, No_ADA_synth_feature, No_ADA_real_feature, sample_num=1000):
    print("start doing classification")
    synth_sample_index = np.random.choice(ADA_synth_feature.shape[0], sample_num)
    real_sample_index = np.random.choice(ADA_real_feature.shape[0], sample_num)
    ADA_synth_feature_sample = ADA_synth_feature[synth_sample_index]
    ADA_real_feature_sample = ADA_real_feature[real_sample_index]
    ADA_feature = np.concatenate((ADA_synth_feature_sample, ADA_real_feature_sample), axis=0)

    No_ADA_synth_feature_sample = No_ADA_synth_feature[synth_sample_index]
    No_ADA_real_feature_sample = No_ADA_real_feature[real_sample_index]
    No_ADA_feature = np.concatenate((No_ADA_synth_feature_sample, No_ADA_real_feature_sample), axis=0)
    target = np.concatenate((np.ones(len(No_ADA_synth_feature_sample)), np.zeros(len(No_ADA_real_feature_sample))), axis=0)

    # use cross-validation
    classifier = SVC()
    ADA_result = cross_val_score(classifier, ADA_feature, target, cv=5)
    No_ADA_result = cross_val_score(classifier, No_ADA_feature, target, cv=5)
    print("ADA score is {}".format(sum(ADA_result)/len(ADA_result)))
    print("No ADA score is {}".format(sum(No_ADA_result)/len(No_ADA_result)))


if __name__ == '__main__':
    ADA_path = # embedded feature dir using domain adaptation
    ADA_synth_path = os.path.join(ADA_path, "synth")
    ADA_real_path = os.path.join(ADA_path, "strong")
    No_ADA_path = # embedded feature dir without using domain adaptation
    No_ADA_synth_path =  os.path.join(No_ADA_path, "synth")
    No_ADA_real_path =  os.path.join(No_ADA_path, "strong")

    # standardize scaler
    scaler = StandardScaler()

    # load and concatenate ADA synthetic features
    for feature_path in os.listdir(ADA_synth_path):
        if feature_path == os.listdir(ADA_synth_path)[0]:
            ADA_synth_feature = np.load(os.path.join(ADA_synth_path, feature_path))
        else:
            ADA_synth_feature = np.concatenate((ADA_synth_feature, np.load(os.path.join(ADA_synth_path, feature_path))), axis=0)
    ADA_synth_feature = ADA_synth_feature.reshape(ADA_synth_feature.shape[0]*ADA_synth_feature.shape[1], ADA_synth_feature.shape[2])

    # load and concatenate ADA real features
    for feature_path in os.listdir(ADA_real_path):
        if feature_path == os.listdir(ADA_real_path)[0]:
            ADA_real_feature = np.load(os.path.join(ADA_real_path, feature_path))
        else:
            ADA_real_feature = np.concatenate((ADA_real_feature, np.load(os.path.join(ADA_real_path, feature_path))), axis=0)
    ADA_real_feature = ADA_real_feature.reshape(ADA_real_feature.shape[0]*ADA_real_feature.shape[1], ADA_real_feature.shape[2])

    
    # load and concatenate none ADA synthetic features
    for feature_path in os.listdir(No_ADA_synth_path):
        if feature_path == os.listdir(No_ADA_synth_path)[0]:
            No_ADA_synth_feature = np.load(os.path.join(No_ADA_synth_path, feature_path))
        else:
            No_ADA_synth_feature = np.concatenate((No_ADA_synth_feature, np.load(os.path.join(No_ADA_synth_path, feature_path))), axis=0)
    No_ADA_synth_feature = No_ADA_synth_feature.reshape(No_ADA_synth_feature.shape[0]*No_ADA_synth_feature.shape[1], No_ADA_synth_feature.shape[2])

    # load and concatenate none ADA real features
    for feature_path in os.listdir(No_ADA_real_path):
        if feature_path == os.listdir(ADA_real_path)[0]:
            No_ADA_real_feature = np.load(os.path.join(No_ADA_real_path, feature_path))
        else:
            No_ADA_real_feature = np.concatenate((No_ADA_real_feature, np.load(os.path.join(No_ADA_real_path, feature_path))), axis=0)
    No_ADA_real_feature = No_ADA_real_feature.reshape(No_ADA_real_feature.shape[0]*No_ADA_real_feature.shape[1], No_ADA_real_feature.shape[2])

    print('finish loading')

    
    # visualize with tsne
    start = time.time()
    visualization(ADA_synth_feature, ADA_real_feature, No_ADA_synth_feature, No_ADA_real_feature, ADA_path=ADA_path, No_ADA_path=No_ADA_path)
    end = time.time()
    print('finish visualization takes {} s'.format(end-start))
    
    
    # svm classification
    start = time.time()
    svm_classfication(ADA_synth_feature, ADA_real_feature, No_ADA_synth_feature, No_ADA_real_feature)
    end = time.time()
    print('finish svm classification takes {} s'.format(end-start))
    
