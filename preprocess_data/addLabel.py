import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split 
import sys
import time

pd.set_option('mode.use_inf_as_na', True) # convert inf to nan
csvNames = ['Monday-WorkingHours.pcap_ISCX.csv', 'Tuesday-WorkingHours.pcap_ISCX.csv', 'Wednesday-WorkingHours.pcap_ISCX.csv', 
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX-copy.csv', 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'Friday-WorkingHours-Morning.pcap_ISCX.csv', 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']
isMerged = [1, 1, 1, 0, 0, 0, 0, 0]
# absolute_path = "/home/wyx/MachineLearningCSV/MachineLearningCVE/"
absolute_path = "../GeneratedLabelledFlows/TrafficLabelling/"
subfix_pcap = '_Flow.csv'
labels = {}
all_flows = {}

src_filepaths = ['Monday-WorkingHours.pcap', 'Tuesday-WorkingHours.pcap', 'Wednesday-WorkingHours.pcap', 'Thursday-WorkingHours.pcap', 'Friday-WorkingHours.pcap']
packet_filepaths = [fname+'_packet.csv' for fname in src_filepaths]
src_filepaths = [fname+'_Flow.csv' for fname in src_filepaths]

def labeled_data(labeled_list, unlabeled_list):
    print("len_label=%d len_unlabel=%d" %(len(labeled_list), len(unlabeled_list)))
    p = 1; q = 1
    h = 0; hh = 0
    tt={}
    mm = 0
    for i in range(1, len(labeled_list)):
        flow_id = labeled_list[i][0]
        if flow_id not in all_flows:
            all_flows[flow_id] = []
        all_flows[flow_id].append(i)

    for i in range(1, len(unlabeled_list)):
        flow_id = unlabeled_list[i][0]
        src, dst, srcport, dstport, proto = flow_id.split('-')

        if (flow_id not in all_flows) or (len(all_flows[flow_id]) == 0):
            # print("not found " + flow_id)
            flow_id = dst+'-'+src+'-'+dstport+'-'+srcport+'-'+proto

        if (flow_id not in all_flows) or (len(all_flows[flow_id]) == 0):
            # print("not found " + flow_id)
            hh+=1
            continue
        k = 0

        if (unlabeled_list[i][6].endswith(' PM')):
            aft = 1
        else:
            aft = 0
        tt = time.strptime(unlabeled_list[i][6].strip(' AM').strip(' PM'), "%d/%m/%Y %H:%M:%S")
        flag = False
        # t0 = int(time.mktime(tt))
        while (k < len(all_flows[flow_id])):
            j = all_flows[flow_id][k]
            ll = time.strptime(labeled_list[j][6], "%d/%m/%Y %H:%M:%S")
            # l0 = int(time.mktime(ll))
            if (int(labeled_list[j][8]) == int(unlabeled_list[i][8])) and (int(unlabeled_list[i][9]) == int(labeled_list[j][9])) and\
                (ll.tm_min == tt.tm_min) and ((ll.tm_hour + 2) % 12 + 1 == tt.tm_hour) and (ll.tm_sec == tt.tm_sec):
                    # qq = ll.tm_hour + 3
                    # if qq>13: qq = 1
                    # if (float(labeled_list[j][10]) != float(unlabeled_list[i][10])) or (float(unlabeled_list[i][11]) != float(labeled_list[j][11])):
                    #     print("warning!!%s %s %s %s" %(labeled_list[j][10], labeled_list[j][11], labeled_list[i][10], labeled_list[i][11]))

                    unlabeled_list[i][-1] = labeled_list[j][-1]
                    del(all_flows[flow_id][k])
                    flag = True
                    # if k>4:
                    #     print(unlabeled_list[i][6] + " " + labeled_list[j][6] + " " + str(k))
                    break
                # print("Serious dismatch with %s which seqid is %d,%d" %(flow_id, i, j))
                # print("number of Packet are %s %s and %s %s" %(labeled_list[j][8], labeled_list[j][9], unlabeled_list[i][8], unlabeled_list[i][9]))
            else:
                k+=1
        if not flag:
            h += 1
            print(flow_id)
        else:
            mm = max(mm, k)

    print("len_label=%d len_unlabel=%d" %(len(labeled_list), len(unlabeled_list)))
    print(h)
    print(hh)
    print(mm)

def main():
    # merge_all_file()
    for i in range(0, 1):
        f = open(src_filepaths[i],"r")
        reader = csv.reader(f)
        unlabeled_list = list(reader)
        f = open(absolute_path+csvNames[i],"r")
        reader = csv.reader(f)
        labeled_list = list(reader)
        # f = open(absolute_path+csvNames[i+1],"r")
        # reader = csv.reader(f)
        # labeled_list += list(reader)
        # f = open(absolute_path+csvNames[7],"r")
        # reader = csv.reader(f)
        # labeled_list += list(reader)

        labeled_data(labeled_list, unlabeled_list)

        f = open('out/'+src_filepaths[i],"w",newline='')
        writer=csv.writer(f,dialect='excel')
        for i in range(len(unlabeled_list)):
            writer.writerow(unlabeled_list[i])
        f.close()

if __name__ == '__main__':
    main()
    # main()
