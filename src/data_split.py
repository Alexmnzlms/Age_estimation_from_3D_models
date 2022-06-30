import pandas as pd
import math
import numpy as np
import ranges_of_age as roa
from collections import Counter
import cv2 as cv
import scipy.ndimage as sci
import functions as fn

pubis_age = pd.read_csv('./pubis.csv')

pubis_age.at[0,'Edad']

with open('./data_img/lsFeaturemap') as f:
    lines = f.readlines()

names = [line[:-1] for line in lines]
names = names[1:]

dic_names = {}
list_number = []
for name in names:
    n = name.split("_")[0]
    if n == '270a' or n == '270b':
        n = '270'
        
    age = pubis_age.at[int(n)-1,'Edad']
    if name not in dic_names.keys():
        dic_names[name] = age
        list_number.append(n)

sample_df = pd.DataFrame(list(dic_names.items()), columns=['Sample', 'Age'])
sample_df.insert(0,"Number",list_number, True)
print(sample_df)

for sample in sample_df.index:
    if math.isnan(sample_df.at[sample,'Age']):
        sample_df = sample_df.drop(sample)

print(sample_df)
sample_df.sort_values(by=['Age'],inplace=True)
print(sample_df)

age_list = sample_df['Age'].to_list()
dic_ranges = dict.fromkeys([fn.bin(label,age_list) for label in age_list],0)
sample_range_list = []

for sample in sample_df.index:
    age = sample_df.at[sample,'Age']
    age = int(age)
    for i,r in enumerate(dic_ranges.keys()):
        if r == fn.bin(age,age_list):
            dic_ranges[r] += 1
            sample_range_list.append(i)
            break

print(dic_ranges)

sample_df.insert(2,'Range', sample_range_list, True)
print(sample_df)

print('\n[INFO] Calculando pesos...\n')

print(age_list)

print(sample_df)
train_val, test = fn.train_test_split(sample_df,0.9)

print(train_val)
print(test)
train, validation = fn.train_test_split(train_val,0.75)

fn.calculate_weights(train)

print(train)
print(validation)
print(test)

train = train.drop(columns=["Range"])
train.to_csv('./train.csv',index=False)
validation = validation.drop(columns=["Range"])
validation.to_csv('./validation.csv',index=False)
test = test.drop(columns=["Range"])
test.to_csv('./test.csv',index=False)
train_val = train_val.drop(columns=["Range","Set"])
train_val.to_csv('./trainval.csv',index=False)

sample_df = sample_df.drop(columns=["Range","Set"])
sample_df.to_csv('./dataset.csv',index=False)
