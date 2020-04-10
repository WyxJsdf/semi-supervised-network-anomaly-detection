import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split 
import sys
import time

pd.set_option('mode.use_inf_as_na', True) # convert inf to nan
# absolute_path = "/home/wyx/MachineLearningCSV/MachineLearningCVE/"
absolute_path = "../GeneratedLabelledFlows/TrafficLabelling/"
subfix_pcap = '_Flow.csv'
labels = {}
all_flows = {}

src_filepaths = ['Monday-WorkingHours.pcap', 'Tuesday-WorkingHours.pcap', 'Wednesday-WorkingHours.pcap', 'Thursday-WorkingHours.pcap', 'Friday-WorkingHours.pcap']
packet_filepaths = [fname+'_Packet.csv' for fname in src_filepaths]
src_filepaths = [fname+'_Flow.csv' for fname in src_filepaths]


def main():
    # merge_all_file()
    fout_stat = open("all_stat.csv","w",newline='')
    writer_stat = csv.writer(fout_stat,dialect='excel')
    h = 0
    h1 = 0
    h2 = 0
    fout_seq = open("all_seq.csv","w",newline='')
    writer_seq = csv.writer(fout_seq,dialect='excel')
    for i in range(0, 5):
        print("now reading done file %d" %(i))
        f = open(src_filepaths[i],"r")
        reader = csv.reader(f)
        stat_list = list(reader)
        f = open(packet_filepaths[i],"r")
        reader = csv.reader(f)
        seq_list = list(reader)

        if i==0:
            writer_stat.writerow(stat_list[0])
        for j in range(1, len(stat_list)):
            if stat_list[j][-1] == 'NeedManualLabel':
                h+=1
                continue
            if stat_list[j][5] != '6':
                h1+=1
                continue
            seq_list[j-1][-1] = stat_list[j][-1]
            if (len(seq_list[j-1]) > 10000 or seq_list[j-1][0] != stat_list[j][0]):
                h2+=1
                continue
            writer_stat.writerow(stat_list[j])
            writer_seq.writerow(seq_list[j-1])
    print(h)
    print(h1)
    print(h2)
    fout_stat.close()
    fout_seq.close()
if __name__ == '__main__':
    main()
    # main()
