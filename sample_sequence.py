import numpy as np
import csv
from sklearn.model_selection import train_test_split 

absolute_path = "../data/IDS2017/all.seq"
output_path = "cicids2017_seq_sample_0.01_500000.csv"

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('outpath', type=str, help='Path of the original embeddings to load')
    parser.add_argument('--epoch', type=int,
        help='number of images in this folder.', default=300)


def sample_all_file(path, neg_samples=495000, pos_samples=5000):
    with open (path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        data_list = list(reader)
    ids_pos = list(filter(lambda x: x[-1]=='0', data_list))
    ids_neg = list(filter(lambda x: x[-1]=='1', data_list))
    print("pos=%d neg=%d\n" %(len(ids_pos), len(ids_neg)))
    
    ids_pos = np.array(ids_pos)
    ids_neg = np.array(ids_neg)
    np.random.shuffle(ids_pos)
    np.random.shuffle(ids_neg)
    ids_pos = ids_pos[:pos_samples]
    ids_neg = ids_neg[:neg_samples]
    with open(output_path,"w", newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows(ids_pos)
        writer.writerows(ids_neg)

def main():
    # merge_all_file()
    sample_all_file(absolute_path)

if __name__ == '__main__':
   # main(parse_arguments(sys.argv[1:]))
    main()
