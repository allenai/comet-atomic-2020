import pandas as pd
from collections import OrderedDict
import csv
import os
import argparse


def main():
    print('\n#######################')
    print('Calculate Coverage')
    print('#######################')



    # OUTPUT DIR
    output_dir = os.path.join(args.data_dir, 'output-x')
    print("Outputting matches to %s"%output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # INPUT FILENAMES
    atomic2020_file = os.path.join(args.data_dir, 'atomic2020_processed.tsv')
    cn_file = os.path.join(args.data_dir, 'conceptnet_processed.tsv')
    atomic_file = os.path.join(args.data_dir, 'atomic_processed.tsv')
    trans_omcs_file = os.path.join(args.data_dir, 'transomcs_processed.tsv')

    print('\nReading atomic2020')
    dict_a2 = read_into_odict(atomic2020_file)
    print('Reading conceptnet')
    dict_cn = read_into_odict(cn_file)
    print('Reading atomic')
    dict_a1 = read_into_odict(atomic_file)
    print('Reading transOMCS')
    dict_to = read_into_odict(trans_omcs_file)

    kbs = [('atomic',dict_a1), ('conceptnet',dict_cn),  ('atomic2020', dict_a2), ('transomcs',dict_to)]
    kbs_vs = [('atomic',dict_a1), ('conceptnet',dict_cn), ('atomic2020', dict_a2), ('transomcs',dict_to)]

    print("\nKB PAIR,TUPLE MATCH COUNTS")

    for (kb1,d1) in kbs:
        for (kb2,d2) in kbs_vs:
            if kb1 == kb2:
                continue

            name = kb1+'-'+kb2
            calculate_hrt(d1, d2, mappings[name], name, output_dir)


def calculate_hrt(d1, d2, relation_mappings, pair_name, output_dir, direction='hrt'):
    hrt_match = []
    hr_match = []

    hrt_match_cnt = 0
    hr_match_cnt = 0

    no_match = []

    for relation1 in d1:
        if relation1 not in relation_mappings:
            continue
        mappings = relation_mappings[relation1]
        for head1, tails1 in d1[relation1].items():
            for relation2 in mappings:
                if relation2 not in d2:
                    continue
                for head2, tails2 in d2[relation2].items():
                    if head2 < head1:
                        continue
                    elif head2 == head1:
                        hr_match_cnt += 1
                        hr = [relation1, relation2, head1, head2]
                        hr_match.append(hr)

                        if direction == 'hrt':
                            for tail1 in tails1:
                                for tail2 in tails2:
                                    if tail2 < tail1:
                                        continue
                                    elif tail2 == tail1:
                                        hrt_match_cnt += 1
                                        hrt_match.append(hr + [tail1, tail2])
                                        break
                                    else:
                                        no_match.append(hr + [tail1, tail2])
                                        break
                    else:
                        break

    print("%s,%s"%(pair_name, hrt_match_cnt))

    with open(os.path.join(output_dir,pair_name+'-'+direction+'-match.csv'),'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(hrt_match)


def read_into_odict(af,direction='hrt'):
    if direction == 'trh':
        index_by = 6
        other_col = 5
    else:
        index_by = 5
        other_col = 6

    df_a = pd.read_csv(af, header=None, index_col=[1, index_by], sep='\t')
    df = df_a.sort_index()
    df.fillna('none',inplace=True)

    return df.groupby(level=0).apply(lambda df: df.sort_index().xs(df.name)[other_col].sort_values(ascending=True).groupby(level=0).agg(list).to_dict(OrderedDict)).to_dict(into=OrderedDict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', help="directory with KBs in tab separated data files and results from preprocess_kb_triples_part2.")

    args = parser.parse_args()

    # KB LABEL MAPPINGS
    atomic_cn_mapping = {
        'ObjectUse': ['UsedFor', 'ReceivesAction'],
        'HasProperty': ['HasProperty', 'HasA'],
        'MadeUpOf': ['PartOf', 'MadeOf', 'HasA', 'ReceivesAction'],
        'AtLocation': ['AtLocation', 'ReceivesAction'],
        'CapableOf': ['CapableOf', 'HasPrerequisite'],
        'Desires': ['Desires'],
        'NotDesires': ['NotDesires'],
        'xIntent': ['MotivatedByGoal'],
        'xReason': ['MotivatedByGoal'],
        'xNeed': ['HasPrerequisite'],
        'xWant': ['CausesDesire'],
        'HasSubEvent': ['HasSubevent', 'HasFirstSubevent', 'HasLastSubevent'],
        'Causes': ['Causes', 'ReceivesAction'],
        'xEffect': ['Causes'],
        'HinderedBy': ['ObstructedBy']
    }

    cn_atomic_mapping = {
        'UsedFor': ['ObjectUse'],
        'ReceivesAction':['Causes','ObjectUse','MadeUpOf','AtLocation'],
        'HasProperty': ['HasProperty'],
        'HasA': ['HasProperty','MadeUpOf'],
        'PartOf': ['MadeUpOf'],
        'MadeOf': ['MadeUpOf'],
        'AtLocation': ['AtLocation'],
        'CapableOf': ['CapableOf'],
        'HasPrerequisite': ['xNeed','CapableOf'],
        'Desires': ['Desires'],
        'NotDesires': ['NotDesires'],
        'MotivatedByGoal': ['xIntent','xReason'],
        'CausesDesire': ['xWant'],
        'HasSubevent': ['HasSubEvent'],
        'HasFirstSubevent': ['HasSubEvent'],
        'HasLastSubevent': ['HasSubEvent'],
        'Causes': ['Causes','xEffect'],
        'ObstructedBy': ['HinderedBy']
    }


    atomic_atomic2020_mapping = {
        'xIntent': ['xIntent'],
        'xNeed': ['xNeed','xReason'],
        'xAttr': ['xAttr'],
        'xReact': ['xReact'],
        'xWant': ['xWant'],
        'xEffect': ['xEffect'],
        'oReact': ['oReact'],
        'oWant': ['oWant'],
        'oEffect': ['oEffect']
    }

    atomic2020_atomic_mapping = {
        'xIntent': ['xIntent'],
        'xNeed': ['xNeed'],
        'xReason': ['xNeed'],
        'xAttr': ['xAttr'],
        'xReact': ['xReact'],
        'xWant': ['xWant'],
        'xEffect': ['xEffect'],
        'oReact': ['oReact'],
        'oWant': ['oWant'],
        'oEffect': ['oEffect']
    }

    conceptnet_labels = {
        'AtLocation':['AtLocation'],
        'CapableOf':['CapableOf'],
        'Causes':['Causes'],
        'CausesDesire':['CausesDesire'],
        'Desires':['Desires'],
        'HasA':['HasA'],
        'HasFirstSubevent':['HasFirstSubevent'],
        'HasLastSubevent':['HasLastSubevent'],
        'HasPrerequisite':['HasPrerequisite'],
        'HasProperty':['HasProperty'],
        'HasSubevent':['HasSubevent'],
        'MadeOf':['MadeOf'],
        'MotivatedByGoal':['MotivatedByGoal'],
        'NotDesires':['NotDesires'],
        'PartOf':['PartOf'],
        'ReceivesAction':['ReceivesAction'],
        'UsedFor':['UsedFor'],
    }

    mappings = {
        'atomic-conceptnet': atomic_cn_mapping,
        'atomic-atomic2020': atomic_atomic2020_mapping,
        'atomic-transomcs': atomic_cn_mapping,
        'conceptnet-atomic': cn_atomic_mapping,
        'conceptnet-atomic2020': cn_atomic_mapping,
        'conceptnet-transomcs': conceptnet_labels,
        'atomic2020-atomic': atomic2020_atomic_mapping,
        'atomic2020-conceptnet': atomic_cn_mapping,
        'atomic2020-transomcs': atomic_cn_mapping,
        'transomcs-atomic': cn_atomic_mapping,
        'transomcs-conceptnet': conceptnet_labels,
        'transomcs-atomic2020': cn_atomic_mapping,
    }


    main()
