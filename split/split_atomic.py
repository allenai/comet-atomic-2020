import argparse
import random

from utils import read_csv, write_array2tsv


def load_atomic(args):
    random.seed(args.random_seed)

    atomic_split_folder = args.data_folder + "original_split/"

    atomic_file = args.data_folder + args.data_file
    atomic_data = read_csv(atomic_file, delimiter="\t", skip_header=True)

    atomic_train_file = atomic_split_folder + "v4_atomic_trn.csv"
    atomic_train = read_csv(atomic_train_file)
    atomic_dev_file = atomic_split_folder + "v4_atomic_dev.csv"
    atomic_dev = read_csv(atomic_dev_file)
    atomic_test_file = atomic_split_folder + "v4_atomic_tst.csv"
    atomic_test = read_csv(atomic_test_file)

    atomic_train_events = set([l[0] for l in atomic_train])
    atomic_dev_events = set([l[0] for l in atomic_dev])
    atomic_test_events = set([l[0] for l in atomic_test])

    atomic_train = [l for l in atomic_data if l[0] in atomic_train_events]
    atomic_dev = [l for l in atomic_data if l[0] in atomic_dev_events]
    atomic_test = [l for l in atomic_data if l[0] in atomic_test_events]

    if args.sanity_check:
        nb_train = 0
        nb_dev = 0
        nb_test = 0
        nb_other = 0
        for d in atomic_data:
            event = d[0]
            if event in atomic_train_events:
                nb_train += 1
            else:
                if event in atomic_dev_events:
                    nb_dev += 1
                else:
                    if event in atomic_test_events:
                        nb_test += 1
                    else:
                        nb_other += 1
        assert nb_other == 0

    return atomic_train, atomic_dev, atomic_test


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, help='Path to folder containing the data',
                        default="./data/atomic/")
    parser.add_argument('--data-file', type=str, help='Dataset filename', default="atomic_v1.tsv")
    parser.add_argument('--random-seed', type=int, default=30, help='Random seed')
    parser.add_argument('--sanity-check', action='store_true',
                        help='If specified, perform sanity check during split creation')
    args = parser.parse_args()

    # Load ATOMIC data
    (train, dev, test) = load_atomic(args)

    # Write tsv files
    folder = args.data_folder
    write_array2tsv(folder + "train.tsv", train)
    write_array2tsv(folder + "dev.tsv", dev)
    write_array2tsv(folder + "test.tsv", test)


if __name__ == "__main__":
    main()
