import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from collections import Counter
import scipy.ndimage as sci
import cv2 as cv

def bin(age, age_list):
    unique_list = []
    for i in np.arange(np.max(age_list)+1):
        if i not in unique_list:
            unique_list.append(i)

    return unique_list.index(age)

def inverse(x):
    v = np.float32(1/x)
    if v == np.Inf:
        v = np.float32(1)
    return v

def calculate_weights(sample_df):
    age_list = sample_df['Age'].to_list()
    bin_index_per_label = [bin(label,age_list) for label in age_list]
    # print(bin_index_per_label, len(bin_index_per_label))

    N_ranges = max(bin_index_per_label) + 1
    num_samples_of_bin = dict(Counter(bin_index_per_label))
    # print(num_samples_of_bin)
    emp_label_dist = [num_samples_of_bin.get(i,0) for i in np.arange(N_ranges)]
    print(emp_label_dist, len(emp_label_dist))
    lds_kernel = cv.getGaussianKernel(5,2).flatten()
    eff_label_dist = sci.convolve1d(np.array(emp_label_dist), weights=lds_kernel, mode='constant')
    print(eff_label_dist, len(eff_label_dist))
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    # print(eff_num_per_label)
    weights = [inverse(x) for x in eff_num_per_label]
    # weights = [1.0 for x in eff_num_per_label]

    sample_df['Weight'] = weights


df = pd.read_csv('./dataset.csv')

print(df)

kf = KFold(n_splits=5,shuffle=True)

for train_index, test_index in kf.split(df):
    # print("\nTRAIN:", train_index, "\nTEST:", test_index)
    train = df.iloc[train_index].copy()
    test = df.iloc[test_index].copy()
    calculate_weights(train)
    print(train)
    print(test)