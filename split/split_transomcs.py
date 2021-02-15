import argparse
import random

from utils import read_csv, write_array2tsv, head_based_split


def load_transomcs(args):
    import matplotlib.pyplot as plt

    random.seed(args.random_seed)

    data_file = args.data_folder + args.data_file
    data = read_csv(data_file, delimiter="\t")

    selection = [[l[0], l[1], l[2]] for l in data if float(l[3]) >= args.confidence_threshold
                 and l[1] not in args.excluded_relations]

    if args.sanity_check:
        confs = [float(l[3]) for l in data]
        plt.hist(confs, density=False, bins=30)
        plt.yscale("log")
        plt.ylabel('Counts')
        plt.xlabel('Confidence')
        plt.show()

    (train, dev, test) = head_based_split(data=selection,
                                          dev_size=args.dev_size,
                                          test_size=args.test_size,
                                          head_size_threshold=args.head_size_threshold)

    return train, dev, test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-folder', type=str, help='Path to folder containing the data',
                        default="./data/transomcs/")
    parser.add_argument('--data-file', type=str, help='Dataset filename', default="TransOMCS_full.txt")
    parser.add_argument('--random-seed', type=int, default=30, help='Random seed')
    parser.add_argument('--dev-size', type=int, default=10000, help='Dev size')
    parser.add_argument('--test-size', type=int, default=100000, help='Test size')
    parser.add_argument('--head-size-threshold', type=int, default=500, help='Maximum number of tuples a head is involved in, '
                                                                   'in order to be a candidate for the dev/test set')
    parser.add_argument('--confidence-threshold', default=0.5, help='Confidence threshold for transomcs tuple')
    parser.add_argument('--excluded-relations', default=["DefinedAs", "LocatedNear"], help='Relations to exclude')
    parser.add_argument('--sanity-check', action='store_true',
                        help='If specified, perform sanity check during split creation')

    args = parser.parse_args()

    # Load TransOMCS data
    (train, dev, test) = load_transomcs(args)

    # Write tsv files
    folder = args.data_folder
    write_array2tsv(folder + "train.tsv", train)
    write_array2tsv(folder + "dev.tsv", dev)
    write_array2tsv(folder + "test.tsv", test)


if __name__ == "__main__":
    main()
