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

def visualization(DANN_synth_feature, DANN_real_feature, No_DANN_synth_feature, No_DANN_real_feature, DANN_path=None, No_DANN_path=None, sample_num=None):
    print("start visualization")

    DANN_tsne_path = os.path.join(DANN_path, 'tsne(clip).npy')
    N_DANN_tsne_path = os.path.join(No_DANN_path, 'tsne(clip)_DASSL.npy')

    if not os.path.exists(DANN_tsne_path):
        if sample_num != None:
            synth_sample_index = np.random.choice(DANN_synth_feature.shape[0], sample_num)
            real_sample_index = np.random.choice(DANN_real_feature.shape[0], sample_num)
            DANN_synth_feature_sample = DANN_synth_feature[synth_sample_index]
            DANN_real_feature_sample = DANN_real_feature[real_sample_index]
            DANN_feature = np.concatenate((DANN_synth_feature_sample, DANN_real_feature_sample), axis=0)
        else:
            DANN_feature = np.concatenate((DANN_synth_feature, DANN_real_feature), axis=0)
        DANN_feature = scaler.fit_transform(DANN_feature)
        DANN_feature = TSNE(n_components=2).fit_transform(DANN_feature)
        np.save(DANN_tsne_path, DANN_feature)

    if not os.path.exists(N_DANN_tsne_path):
        if sample_num != None:
            No_DANN_synth_feature_sample = No_DANN_synth_feature[synth_sample_index]
            No_DANN_real_feature_sample = No_DANN_real_feature[real_sample_index]
            No_DANN_feature = np.concatenate((No_DANN_synth_feature_sample, No_DANN_real_feature_sample), axis=0)
        else:
            No_DANN_feature = np.concatenate((No_DANN_synth_feature, No_DANN_real_feature), axis=0)
        No_DANN_feature = scaler.fit_transform(No_DANN_feature)
        No_DANN_feature = TSNE(n_components=2).fit_transform(No_DANN_feature)
        np.save(N_DANN_tsne_path, No_DANN_feature)
    
    DANN_feature = np.load(DANN_tsne_path)
    No_DANN_feature = np.load(N_DANN_tsne_path)

    if sample_num != None:
        DANN_synth_feature_sample = DANN_feature[:sample_num]
        DANN_real_feature_sample = DANN_feature[sample_num:]
        No_DANN_synth_feature_sample = No_DANN_feature[:sample_num]
        No_DANN_real_feature_sample = No_DANN_feature[sample_num:]
    else:
        DANN_synth_feature_sample = DANN_feature[:DANN_synth_feature.shape[0]]
        DANN_real_feature_sample = DANN_feature[DANN_synth_feature.shape[0]:]
        No_DANN_synth_feature_sample = No_DANN_feature[:DANN_synth_feature.shape[0]]
        No_DANN_real_feature_sample = No_DANN_feature[DANN_synth_feature.shape[0]:]

    plt.figure()
    plt.scatter(DANN_synth_feature_sample[:,0], DANN_synth_feature_sample[:,1], alpha=0.6, s=10, marker='x', label='synth')
    plt.scatter(DANN_real_feature_sample[:,0], DANN_real_feature_sample[:,1], alpha=0.3, s=10, c='r', label='real')
    plt.legend()
    plt.title('DASSL') # plt.title('Adaptation')
    plt.xlim(-60, 80)
    plt.ylim(-90, 90)
    plt.axis('off')
    plt.savefig(os.path.join(DANN_path, 'DASSL(clip).png'))# plt.savefig(os.path.join(DANN_path, 'Adaptation(clip).png'))

    plt.figure()
    plt.scatter(No_DANN_synth_feature_sample[:,0], No_DANN_synth_feature_sample[:,1], alpha=0.6, s=10, marker='x', label='synth')
    plt.scatter(No_DANN_real_feature_sample[:,0], No_DANN_real_feature_sample[:,1], alpha=0.3, s=10, c='r', label='real')
    plt.legend()
    plt.title('No_DASSL') # plt.title('No_adaptation')
    plt.xlim(-60, 80)
    plt.ylim(-90, 90)
    plt.axis('off')
    plt.savefig(os.path.join(No_DANN_path, 'No_DASSL(clip).png'))# plt.savefig(os.path.join(No_DANN_path, 'No_adaptation(clip).png'))

    # synthetic_tsv_path = '../dataset/metadata/train/synthetic20.tsv'
    # synthetic_df = pd.read_csv(synthetic_tsv_path, sep='\t')
    # # pdb.set_trace()
    # synthetic_df['duration'] = synthetic_df['offset'] - synthetic_df['onset']
    # pdb.set_trace()

def svm_classfication(DANN_synth_feature, DANN_real_feature, No_DANN_synth_feature, No_DANN_real_feature, sample_num=None):
    print("start doing classification")
    if sample_num != None:
        synth_sample_index = np.random.choice(DANN_synth_feature.shape[0], sample_num)
        real_sample_index = np.random.choice(DANN_real_feature.shape[0], sample_num)
        DANN_synth_feature_sample = DANN_synth_feature[synth_sample_index]
        DANN_real_feature_sample = DANN_real_feature[real_sample_index]
    else:
        DANN_synth_feature_sample = DANN_synth_feature
        DANN_real_feature_sample = DANN_real_feature
    DANN_feature = np.concatenate((DANN_synth_feature_sample, DANN_real_feature_sample), axis=0)

    if sample_num != None:
        No_DANN_synth_feature_sample = No_DANN_synth_feature[synth_sample_index]
        No_DANN_real_feature_sample = No_DANN_real_feature[real_sample_index]
    else:
        No_DANN_synth_feature_sample = No_DANN_synth_feature
        No_DANN_real_feature_sample = No_DANN_real_feature
    No_DANN_feature = np.concatenate((No_DANN_synth_feature_sample, No_DANN_real_feature_sample), axis=0)

    target = np.concatenate((np.ones(len(No_DANN_synth_feature_sample)), np.zeros(len(No_DANN_real_feature_sample))), axis=0)
    # np.random.shuffle(target)
    
    # X_train, X_test, y_train, y_test = train_test_split(DANN_feature, target)
    # classifier = SVC()
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # print("DANN result: \n")
    # print(classification_report(y_test, y_pred))

    # X_train, X_test, y_train, y_test = train_test_split(No_DANN_feature, target)
    # classifier = SVC()
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)

    # print("No DANN result: \n")
    # print(classification_report(y_test, y_pred))

    # use cross-validation
    classifier = SVC()
    DANN_result = cross_val_score(classifier, DANN_feature, target, cv=5)
    No_DANN_result = cross_val_score(classifier, No_DANN_feature, target, cv=5)
    print("DANN score is {}".format(sum(DANN_result)/len(DANN_result)))
    print("No DANN score is {}".format(sum(No_DANN_result)/len(No_DANN_result)))

def z_test(DANN_synth_feature, DANN_real_feature, No_DANN_synth_feature, No_DANN_real_feature):
    print("start doing z-test")
    DANN_ztest_list = []
    No_DANN_ztest_list = []
    idx = np.random.randint(len(DANN_synth_feature), size=len(DANN_real_feature))
    DANN_synth_feature = DANN_synth_feature[idx, :]
    No_DANN_synth_feature = No_DANN_synth_feature[idx, :]
    DANN_tstate, DANN_pvalue = ztest(DANN_synth_feature, DANN_real_feature)
    No_DANN_tstate, No_DANN_pvalue = ztest(No_DANN_synth_feature, No_DANN_real_feature)
    print('Z-statistic of DANN is {}'.format(DANN_pvalue))
    print('Z-statistic of No DANN is {}'.format(No_DANN_pvalue))
    pdb.set_trace()

if __name__ == '__main__':
    DANN_path = "feature_output/MeanTeacher_with_synthetic_DAfirst_shift_ICT_pseudolabel_2_pd/"
    DANN_synth_path = os.path.join(DANN_path, "synth")
    DANN_real_path = os.path.join(DANN_path, "strong")
    No_DANN_path = "feature_output/MeanTeacher_with_synthetic_2_nomeanteacher/"
    No_DANN_synth_path =  os.path.join(No_DANN_path, "synth")
    No_DANN_real_path =  os.path.join(No_DANN_path, "strong")

    # standardize scaler
    scaler = StandardScaler()

    # load and concatenate DANN synthetic features
    for feature_path in os.listdir(DANN_synth_path):
        if feature_path == os.listdir(DANN_synth_path)[0]:
            DANN_synth_feature = np.load(os.path.join(DANN_synth_path, feature_path))
        else:
            DANN_synth_feature = np.concatenate((DANN_synth_feature, np.load(os.path.join(DANN_synth_path, feature_path))), axis=0)
    DANN_synth_feature = DANN_synth_feature.reshape(DANN_synth_feature.shape[0], DANN_synth_feature.shape[1]*DANN_synth_feature.shape[2])

    # load and concatenate DANN real features
    for feature_path in os.listdir(DANN_real_path):
        if feature_path == os.listdir(DANN_real_path)[0]:
            DANN_real_feature = np.load(os.path.join(DANN_real_path, feature_path))
        else:
            DANN_real_feature = np.concatenate((DANN_real_feature, np.load(os.path.join(DANN_real_path, feature_path))), axis=0)
    DANN_real_feature = DANN_real_feature.reshape(DANN_real_feature.shape[0], DANN_real_feature.shape[1]*DANN_real_feature.shape[2])
    
    
    # load and concatenate none DANN synthetic features
    for feature_path in os.listdir(No_DANN_synth_path):
        if feature_path == os.listdir(No_DANN_synth_path)[0]:
            No_DANN_synth_feature = np.load(os.path.join(No_DANN_synth_path, feature_path))
        else:
            No_DANN_synth_feature = np.concatenate((No_DANN_synth_feature, np.load(os.path.join(No_DANN_synth_path, feature_path))), axis=0)
    No_DANN_synth_feature = No_DANN_synth_feature.reshape(No_DANN_synth_feature.shape[0], No_DANN_synth_feature.shape[1]*No_DANN_synth_feature.shape[2])

    # load and concatenate none DANN real features
    for feature_path in os.listdir(No_DANN_real_path):
        if feature_path == os.listdir(DANN_real_path)[0]:
            No_DANN_real_feature = np.load(os.path.join(No_DANN_real_path, feature_path))
        else:
            No_DANN_real_feature = np.concatenate((No_DANN_real_feature, np.load(os.path.join(No_DANN_real_path, feature_path))), axis=0)
    No_DANN_real_feature = No_DANN_real_feature.reshape(No_DANN_real_feature.shape[0], No_DANN_real_feature.shape[1]*No_DANN_real_feature.shape[2])

    print('finish loading')

    
    # visualize with tsne
    start = time.time()
    visualization(DANN_synth_feature, DANN_real_feature, No_DANN_synth_feature, No_DANN_real_feature, DANN_path=DANN_path, No_DANN_path=No_DANN_path)
    end = time.time()
    print('finish visualization takes {} s'.format(end-start))
    
    
    
    # svm classification
    start = time.time()
    svm_classfication(DANN_synth_feature, DANN_real_feature, No_DANN_synth_feature, No_DANN_real_feature)
    end = time.time()
    print('finish svm classification takes {} s'.format(end-start))

    # # z test
    # z_test(DANN_synth_feature, DANN_real_feature, No_DANN_synth_feature, No_DANN_real_feature)
