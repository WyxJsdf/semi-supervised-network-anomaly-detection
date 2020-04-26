import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split 
import sys
import time

pd.set_option('mode.use_inf_as_na', True) # convert inf to nan
# absolute_path = "/home/wyx/MachineLearningCSV/MachineLearningCVE/"
absolute_path = "../GeneratedLabelledFlows/TrafficLabelling/"
filename = 'all_stat_udp.csv'
labels = {}
all_flows = {}

def revert_csv(df):
    df['Flow Bytes/s']=df['Flow Bytes/s'].astype('float64')
    df['Flow Packets/s']=df['Flow Packets/s'].astype('float64')
    df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].mean(),inplace=True)
    df['Flow Packets/s'].fillna(df['Flow Packets/s'].mean(),inplace=True)
    for index, row in df.iterrows():
        if row['Label'] in labels:
            labels[row['Label']] += 1
        else:
            labels[row['Label']] = 1
    df['RawLabel'] = df['Label']
    df['Label'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)
    try:
        df = df.drop(columns=['Flow ID','Src IP','Src Port','Dst IP','Protocol','Timestamp'])
    except:
        pass
    return df


def parse_csv(filename, revert_flag=True):
    print("Now parsing the csv file: " + filename)
    df=pd.read_csv(filename, header=0,low_memory=False)
    if revert_flag:
        df = revert_csv(df)
    return df


def main():
    df = parse_csv(filename, True)
    df.to_csv("clean_"+filename, index=False)
    # merge_all_file()
if __name__ == '__main__':
    main()
    # main()
