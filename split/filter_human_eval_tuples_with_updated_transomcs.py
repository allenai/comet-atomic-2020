from utils import read_csv, write_tsv


def tuple_key(d):
    return d[0] + d[1] + d[2]


def main():

    folder = "./data/transomcs/"
    file = folder + "TransOMCS_full.txt"
    data = read_csv(file, delimiter="\t")

    confidences = {}
    for d in data:
        key = tuple_key(d)
        confidences[key] = float(d[3])

    human_eval_file = folder + "human_evaluation_tuples.tsv"
    tuples = read_csv(human_eval_file, delimiter="\t", skip_header=True)

    updated_t = [{"head_event": t[0], "relation": t[1], "tail_event": t[2]} for t in tuples if confidences[tuple_key(t)] >= 0.5]
    dropped = [{"head_event": t[0], "relation": t[1], "tail_event": t[2]} for t in tuples if confidences[tuple_key(t)] < 0.5]

    output_file = folder + "human_evaluation_tuples_v2.tsv"
    write_tsv(output_file, updated_t)

    output_file = folder + "dropped_human_evaluation_tuples_v2.tsv"
    write_tsv(output_file, dropped)



if __name__ == "__main__":
    main()