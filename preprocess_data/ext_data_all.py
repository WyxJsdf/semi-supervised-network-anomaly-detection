import numpy as np
import csv
import sys
import argparse

absolute_path = "../../data/IDS2017/flows/out/"
seq_filename = "clean_all_seq_udp_1.csv"
stat_filename = "clean_all_stat_udp.csv"
output_seq_path = "all_ids2017_seq_sample"
output_stat_path = "all_ids2017_stat_sample"


labels_stat = {}
labels_seq = {}
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float,
        help='ratio of the anomaly data', default=0.1)
    return parser.parse_args(argv)

def sample_all_file(all_samples=500000, anomaly_ratio=0.01):
    pos_samples = int(all_samples * anomaly_ratio)
    neg_samples = all_samples - pos_samples
    with open(absolute_path+seq_filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        seq_list = list(reader)
    with open(absolute_path+stat_filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        stat_list = list(reader)

    stat_header = stat_list[0]
    del(stat_list[0])

    ids_pos = [i for i in range(len(seq_list)) if seq_list[i][-2]=='1']
    ids_neg = [i for i in range(len(seq_list)) if seq_list[i][-2]=='0']
    print("pos=%d neg=%d\n" %(len(ids_pos), len(ids_neg)))
    
    ids_pos = np.array(ids_pos)
    ids_neg = np.array(ids_neg)
    np.random.shuffle(ids_pos)
    np.random.shuffle(ids_neg)
    ids_pos = ids_pos[:pos_samples]
    ids_neg = ids_neg[:neg_samples]

    path_seq = output_seq_path + '_' + str(anomaly_ratio) + '_' + str(all_samples) + '.csv'
    path_stat = output_stat_path + '_' + str(anomaly_ratio) + '_' + str(all_samples) + '.csv'

    fout_stat = open(path_stat,"w",newline='')
    writer_stat = csv.writer(fout_stat,dialect='excel')
    fout_seq = open(path_seq,"w",newline='')
    writer_seq = csv.writer(fout_seq,dialect='excel')

    is_select = {};
    for i in ids_pos:
        is_select[i]=1
    for i in ids_neg:
        is_select[i]=1
    writer_stat.writerow(stat_header)
    for i in range(len(seq_list)):
        if (i in is_select):
            writer_seq.writerow(seq_list[i])
            writer_stat.writerow(stat_list[i])
            if seq_list[i][-1] not in labels_seq:
                labels_seq[seq_list[i][-1]] = 0
            if stat_list[i][-1] not in labels_stat:
                labels_stat[stat_list[i][-1]] = 0
            labels_stat[stat_list[i][-1]] += 1
            labels_seq[seq_list[i][-1]] += 1
    fout_stat.close()
    fout_seq.close()    

def main(args):
    # merge_all_file()
    sample_all_file(500000, args.ratio)
    print(labels_stat)
    print(labels_seq)
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    # main()
