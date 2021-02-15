import argparse
import random

from utils import read_csv, write_jsonl


def sample_kg(args):
    random.seed(args.random_seed)

    data_file = args.input_file
    data = read_csv(data_file, delimiter="\t")

    prefixes = {}

    for l in data:
        prefix = l[0] + " " + l[1]
        if prefix not in prefixes.keys():
            prefixes[prefix] = {"head": l[0], "relation":l[1], "tails": []}
        prefixes[prefix]["tails"].append(l[2])

    excluded_relations = [
        "HasPainIntensity",
        "LocatedNear",
        "LocationOfAction",
        "DesireOf",
        "NotMadeOf",
        "InheritsFrom",
        "InstanceOf",
        "RelatedTo",
        "SymbolOf",
        "CreatedBy",
        "NotHasA",
        "NotIsA",
        "NotHasProperty",
        "NotCapableOf",
        "IsA",
        "DefinedAs"
    ]

    print(len(list(prefixes.keys())))
    rel_prefixes = [p for p in prefixes.keys() if prefixes[p]["relation"] not in excluded_relations]
    print(len(rel_prefixes))

    sampled_prefixes = random.sample(list(prefixes.keys()), args.sample_size)

    samples = [prefixes[k] for k in sampled_prefixes]

    rel_samples = [s for s in samples if s["relation"] not in excluded_relations]
    print(len(rel_samples))

    return samples


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-file', type=str, help='Dataset filename')
    parser.add_argument('--output-file', type=str, help='Dataset filename')
    parser.add_argument('--random-seed', type=int, default=30, help='Random seed')
    parser.add_argument('--sample-size', type=int, default=5000, help='Dev size')

    args = parser.parse_args()

    # Load KG data
    samples = sample_kg(args)

    # Write tsv files
    write_jsonl(args.output_file, samples)


if __name__ == "__main__":
    main()
