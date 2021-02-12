from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import argparse
import os

exact2tokenized = {}
tokenized2pos = {}
pos2content = {}


def main():
    print('\n#######################')
    print('Preprocess Part 2')
    print('#######################')

    data_dir = [(os.path.join(args.data_dir,'atomic2020_exact.tsv'),'atomic2020'),
                (os.path.join(args.data_dir,'conceptnet_exact.tsv'),'conceptnet'),
                (os.path.join(args.data_dir,'transomcs_exact.tsv'),'transomcs'),
                (os.path.join(args.data_dir,'atomic_exact.tsv'),'atomic')
    ]

    sw = set(stopwords.words('english'))
    sw.add('x')
    sw.add('y')
    sw.add('z')
    ltz = WordNetLemmatizer()

    for f, kb in data_dir:
        print('{}: processing file'.format(kb))
        d = '/'.join(f.split('/')[:-1]) + '/'
        with open(f) as fin, open(d+kb+'_processed.tsv','w') as fout:
            reader = csv.reader(fin, delimiter='\t')
            writer = csv.writer(fout, delimiter='\t')
            lcnt = 0
            for line in reader:
                if lcnt == 0:  # skipping header
                    lcnt = 1
                    continue

                out_line = process(line, sw, ltz)
                writer.writerow(out_line)


def process(line, sw, ltz):
    head = line[3]
    head_tokens = convert_to_tokens(head)
    head_pos = convert_to_pos(head_tokens)
    head_content = convert_to_content(head_pos, sw, ltz)

    tail = line[4]
    tail_tokens = convert_to_tokens(tail)
    tail_pos = convert_to_pos(tail_tokens)
    tail_content = convert_to_content(tail_pos, sw, ltz)

    return line + [head_content, tail_content]


def convert_to_tokens(exact):
    if exact in exact2tokenized:
        return exact2tokenized[exact]
    else:
        return tokenize(exact)


def convert_to_pos(tokens):
    str_x = list2str(tokens)
    if str_x in tokenized2pos:
        return tokenized2pos[str_x]
    else:
        return postag(tokens)


def convert_to_content(pos, sw, ltz):
    str_x = list_of_tuple2str(pos)
    if str_x in pos2content:
        return pos2content[str_x]
    else:
        return get_content_words(pos, sw, ltz)


def get_content_words(pos_tagged, sw, ltz):
    save = []
    stop_words = []
    for (word, pos) in pos_tagged:
        if word in sw:
            stop_words.append(word)
            continue

        if pos[:2] == 'NN':
            lemmatized = ltz.lemmatize(word, pos="n")
        elif pos[:2] == 'VB':
            lemmatized = ltz.lemmatize(word, pos="v")
        else:
            lemmatized = word

        save.append(lemmatized)

    if len(save) == 0:
        rv = '|'.join(stop_words)
    elif len(save) == 1:
        rv =  save[0]
    else:
        rv = '|'.join(save)
    pos2content[list_of_tuple2str(pos_tagged)] = rv
    return rv


def postag(tokenized):
    p = pos_tag(tokenized)
    tokenized2pos[list2str(tokenized)] = p
    return p


def tokenize(exact):
    t = word_tokenize(exact)
    exact2tokenized[exact] = t
    return t


def list_of_tuple2str(lot):
    return ",".join("(%s,%s)" % tup for tup in lot)


def list2str(l):
    return ",".join(l)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', help="directory with KBs in tab separated data files and results from preprocess_kb_triples_part1")

    args = parser.parse_args()

    main()