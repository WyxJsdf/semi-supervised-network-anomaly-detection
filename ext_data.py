import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

pd.set_option('mode.use_inf_as_na', True) # convert inf to nan
csvNames = ['Monday-WorkingHours.pcap_ISCX.csv', 'Tuesday-WorkingHours.pcap_ISCX.csv', 'Wednesday-WorkingHours.pcap_ISCX.csv', 
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX-copy.csv', 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'Friday-WorkingHours-Morning.pcap_ISCX.csv', 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv']
isMerged = [1, 1, 1, 1, 1, 1, 1, 1]
# absolute_path = "/home/wyx/MachineLearningCSV/MachineLearningCVE/"
absolute_path = "../data/IDS2017/GeneratedLabelledFlows/TrafficLabelling/"
output_path = "cicids2017_all_data.csv"
output_path = "cicids2017_sample_0.01_500000.csv"
labels = {}

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('outpath', type=str, help='Path of the original embeddings to load')
    parser.add_argument('--epoch', type=int,
        help='number of images in this folder.', default=300)

def revert_csv(df):
    df['Flow Bytes/s']=df['Flow Bytes/s'].astype('float64')
    df[' Flow Packets/s']=df[' Flow Packets/s'].astype('float64')
    df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].mean(),inplace=True)
    df[' Flow Packets/s'].fillna(df[' Flow Packets/s'].mean(),inplace=True)
    for index, row in df.iterrows():
        if row[' Label'] in labels:
            labels[row[' Label']] += 1
        else:
            labels[row[' Label']] = 1
    df['RawLabel'] = df[' Label']
    df[' Label'] = df[' Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)
    try:
        df = df.drop(columns=['Flow ID',' Source IP',' Source Port',' Destination IP',' Protocol',' Timestamp'])
    except:
        pass
    return df

def sample_rate_pos(df, sampleRate=0.1):
    df_pos = df[df[' Label'] == 1]
    df_neg = df[df[' Label'] == 0]
    df_pos = df_pos.sample(frac = sampleRate)
    return (df_pos.append(df_neg))

def sample_number(df, pos_samples, neg_samples):
    df_pos = df[df[' Label'] == 1]
    df_neg = df[df[' Label'] == 0]
    df_pos = df_pos.sample(n=pos_samples)
    df_neg = df_neg.sample(n=neg_samples)
    return (df_pos.append(df_neg))

def parse_csv(filename, revert_flag=True, sample_flag=False):
    print("Now parsing the csv file: " + filename)
    df=pd.read_csv(absolute_path + filename, header=0,low_memory=False)
    if revert_flag:
        df = revert_csv(df)
    if sample_flag:
        df = sample_pos(df, sampleRate=0.1)
    return df

def merge_all_file():
    flag = True
    for i in range(0, len(csvNames)):
        if isMerged[i]:
            df = parse_csv(csvNames[i], sample_flag=False)
            if flag:
               df.to_csv(output_path, index=False)
               flag = False
            else:
               df.to_csv(output_path, index=False, header=False, mode='a+')

def sample_all_file(neg_samples=495000, pos_samples=5000):
    flag = True
    for i in range(0, len(csvNames)):
        if isMerged[i]:
            if flag:
                df = parse_csv(csvNames[i])
                flag = False
            else:
                df = df.append(parse_csv(csvNames[i]))
            print(df.shape)
    df = sample_number(df, pos_samples=pos_samples, neg_samples=neg_samples)
    df = df.sample(frac=1)
    df.to_csv(output_path, index=False)

def main():
    # merge_all_file()
    sample_all_file()
    print(labels)

if __name__ == '__main__':
#    main(parse_arguments(sys.argv[1:]))
    main()
