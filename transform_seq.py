# -*- coding: UTF-8 -*-
import csv, math, os
from multiprocessing import Pool

#多进程版本

ben_label = 0
mal_label = 1


def get_pllength_from_tuple(tuple_str):
    direction, pllength, time_stamp, TCP_winsize = tuple_str[1:-1].split('|')
    return pllength

def get_all_from_tuple(tuple_str):
    dst_port, hlen, plen, time_stamp, TCP_winsize = tuple_str.split('|')
    
    return (dst_port, hlen+plen, time_stamp, TCP_winsize)

def time_stamp_s2f(tsstr):
    return int(tsstr)

def metadata_to_str(metadata_seq):
    changed_matadata_seq = [0]*len(metadata_seq)
    pre_ts = time_stamp_s2f(metadata_seq[0][2])
    for i in range(len(metadata_seq)):
        dst_port, pllength, time_stamp, TCP_winsize = metadata_seq[i]

        # TODO timestamp transform
        now_ts = time_stamp_s2f(time_stamp)

        delta_ts = now_ts - pre_ts
        pre_ts = now_ts

        changed_matadata_seq[i] = (dst_port, pllength, str(delta_ts), TCP_winsize)

    seq = [ '|'.join(line) for line in changed_matadata_seq]

    return seq



def transdata_part(src_list):
    dst_list = [0] * len(src_list)
    
    for i in range(len(src_list)):
        src_line = src_list[i]
        if src_line[-1] == 'BENIGN':
            label = ben_label
        else:
            label = mal_label

        # src_port = src_line[1]
        # dst_port = src_line[2]
        
        metadata_seq = [ get_all_from_tuple(tuple_str) for tuple_str in src_line[1:-1]]

        seq = metadata_to_str(metadata_seq)

        #print(seq[5:10])

        seq.append(label)
        seq.append(src_line[-1])

        dst_list[i] = seq

    
    return dst_list


def transdata(src_csv_file_path, dst_csv_file_path, process_num = 64, mode_add = False):
    # 提取原csv中所有序列，并分片
    with open(src_csv_file_path, 'r') as f:
        reader = csv.reader(f)
        seq_list = list(reader)
    seq_num = len(seq_list)
    part_len = math.ceil(seq_num / process_num)
    seqs_feed = [seq_list[i:i + part_len] for i in range(0, seq_num, part_len)]

    if (process_num > len(seqs_feed)):
        process_num = len(seqs_feed)
    # 多进程提取序列
    pool = Pool(process_num)
    results = []
    for i in range(process_num):
        results.append(pool.apply_async(transdata_part, (seqs_feed[i], )))
    pool.close()
    pool.join()

    # 拼接多进程结果
    data_list = []
    for respart in results:
        part_list = respart.get()
        #print(type(data_dict))
        #print(data_dict.keys)
        data_list += part_list
    
    lens = [len(line) for line in data_list]
    if mode_add:
        open_mode = 'a'
    else:
        open_mode = 'w'
    with open(dst_csv_file_path, open_mode) as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data_list)

    print('seq data got from file [%s]\n%d samples\nmax_seq_len = %d\nmin_seq_len = %d'%(src_csv_file_path, len(data_list), max(lens), min(lens)))    


if __name__ == "__main__":

    process_num = 32

    
    src_filepath = "all_seq.csv"

    dst_filepath = 'clean_all_seq.csv'

    transdata(src_filepath, dst_filepath, process_num, mode_add=False)

    #src_filepaths = ['cleanFriday.csv']
    #dst_filepaths = ['all01.seq']

    #for src_filepath,dst_filepath in zip(src_filepaths, dst_filepaths):
    #    transdata(src_filepath, dst_filepath, process_num)

    #print(api_preprocess('abbbaaabbbaaabbbaabbaabbaabbaa'))

