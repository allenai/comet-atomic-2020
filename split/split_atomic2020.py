import argparse
import random

from utils import read_csv, write_array2tsv, head_based_split, get_head_set


def load_atomic2020(args):
    random.seed(args.random_seed)

    atomic2020_v1_file = args.data_folder + "atomic_original_tuples.tsv"
    atomic2020_addl_file = args.data_folder + "atomic_additional_tuples.tsv"
    atomic2020_cn_file = args.data_folder + "atomic_conceptnet_tuples.tsv"

    v1_data = read_csv(atomic2020_v1_file, delimiter="\t", skip_header=True)
    addl_data = read_csv(atomic2020_addl_file, delimiter="\t", skip_header=True)
    cn_data_with_id = read_csv(atomic2020_cn_file, delimiter="\t", skip_header=True)
    cn_data = [l[1:] for l in cn_data_with_id]

    # Atomic split
    atomic_train = read_csv(args.atomic_split + "train.tsv", delimiter="\t", skip_header=False)
    atomic_dev = read_csv(args.atomic_split + "dev.tsv", delimiter="\t", skip_header=False)
    atomic_test = read_csv(args.atomic_split + "test.tsv", delimiter="\t", skip_header=False)
    atomic_train_events = get_head_set(atomic_train)
    atomic_dev_events = get_head_set(atomic_dev)
    atomic_test_events = get_head_set(atomic_test)
    v1_data_train = [l for l in v1_data if l[0] in atomic_train_events]
    v1_data_dev = [l for l in v1_data if l[0] in atomic_dev_events]
    v1_data_test = [l for l in v1_data if l[0] in atomic_test_events]
    assert len(v1_data) == len(v1_data_train) + len(v1_data_dev) + len(v1_data_test)

    # CN split
    cn_train = read_csv(args.conceptnet_split + "train.tsv", delimiter="\t", skip_header=False)
    cn_dev = read_csv(args.conceptnet_split + "dev.tsv", delimiter="\t", skip_header=False)
    cn_test = read_csv(args.conceptnet_split + "test.tsv", delimiter="\t", skip_header=False)
    cn_train_heads = get_head_set(cn_train)
    cn_dev_heads = get_head_set(cn_dev)
    cn_test_heads = get_head_set(cn_test)

    cn_data_train = [l for l in cn_data if l[0] in cn_train_heads]
    cn_data_dev = [l for l in cn_data if l[0] in cn_dev_heads]
    cn_data_test = [l for l in cn_data if l[0] in cn_test_heads]

    # Additional tuples split
    (addl_train, addl_dev, addl_test) = head_based_split(addl_data,
                                                         dev_size=args.dev_size,
                                                         test_size=args.test_size,
                                                         head_size_threshold=args.head_size_threshold,
                                                         dev_heads=atomic_dev_events,
                                                         test_heads=atomic_test_events)

    new_addl_train = []
    new_addl_dev = []
    new_addl_test = addl_test
    for l in addl_train:
        h = l[0]
        if h in cn_dev_heads:
            new_addl_dev.append(l)
        else:
            if h in cn_test_heads:
                new_addl_test.append(l)
            else:
                new_addl_train.append(l)

    for l in addl_dev:
        h = l[0]
        if h in cn_test_heads:
            new_addl_test.append(l)
        else:
            new_addl_dev.append(l)

    train = v1_data_train + cn_data_train + new_addl_train
    dev = v1_data_dev + cn_data_dev + new_addl_dev
    test = v1_data_test + cn_data_test + new_addl_test

    return train, dev, test


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, help='Path to folder containing the data',
                        default="./data/atomic2020/")
    parser.add_argument('--atomic-split', type=str, help='Path to folder containing the ATOMIC split',
                        default="./data/atomic/")
    parser.add_argument('--conceptnet-split', type=str, help='Path to folder containing the ConceptNet split',
                        default="./data/conceptnet/")
    parser.add_argument('--data-file', type=str, help='Dataset filename', default="atomic_v1.tsv")
    parser.add_argument('--dev-size', type=int, default=20000, help='Dev size')
    parser.add_argument('--test-size', type=int, default=50000, help='Test size')
    parser.add_argument('--head-size-threshold', type=int, default=500, help='Maximum number of tuples a head is involved in, '
                                                                   'in order to be a candidate for the dev/test set')
    parser.add_argument('--random-seed', type=int, default=30, help='Random seed')
    parser.add_argument('--sanity-check', action='store_true',
                        help='If specified, perform sanity check during split creation')
    args = parser.parse_args()

    # Load ATOMIC 2020 data
    (train, dev, test) = load_atomic2020(args)

    # Write tsv files
    folder = args.data_folder
    write_array2tsv(folder + "train.tsv", train)
    write_array2tsv(folder + "dev.tsv", dev)
    write_array2tsv(folder + "test.tsv", test)


if __name__ == "__main__":
    main()
