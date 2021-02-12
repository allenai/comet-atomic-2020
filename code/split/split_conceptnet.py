import argparse
import random

from utils import read_csv, write_array2tsv, head_based_split, count_relation


def load_conceptnet(args):
    random.seed(args.random_seed)

    cn_train_file = args.data_folder + args.data_file
    cn_train_data = read_csv(cn_train_file, delimiter="\t")
    train_data = [[l[1], l[0], l[2]] for l in cn_train_data]
    count_relation(train_data)
    original_test_data = []
    if args.include_original_test:
        cn_test_file = args.data_folder + "test.txt"
        cn_test_data = read_csv(cn_test_file, delimiter="\t")
        original_test_data = [[l[1], l[0], l[2]] for l in cn_test_data if float(l[3]) == 1.0]
        if args.sanity_check:
            assert len(original_test_data) == 1200

    (train, dev, test) = head_based_split(train_data, args.dev_size, args.test_size, args.head_size_threshold)

    return train, dev, original_test_data + test


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, help='Path to folder containing the data',
                        default="./data/conceptnet/")
    parser.add_argument('--data-file', type=str, help='Dataset filename', default="train300k.txt")
    parser.add_argument('--dev-size', type=int, default=5000, help='Dev size')
    parser.add_argument('--test-size', type=int, default=30000, help='Test size')
    parser.add_argument('--head-size-threshold', default=500, help='Maximum number of tuples a head is involved in, '
                                                                   'in order to be a candidate for the dev/test set')
    parser.add_argument('--random-seed', type=int, default=30, help='Random seed')
    parser.add_argument('--sanity-check', action='store_true',
                        help='If specified, perform sanity check during split creation')
    parser.add_argument('--include-original-test', action='store_true',
                        help='If specified, include the original 1.2k test set')
    args = parser.parse_args()

    # Load ConceptNet data
    (train, dev, test) = load_conceptnet(args)

    # Write tsv files
    folder = args.data_folder
    write_array2tsv(folder + "train.tsv", train)
    write_array2tsv(folder + "dev.tsv", dev)
    write_array2tsv(folder + "test.tsv", test)


if __name__ == "__main__":
    main()
