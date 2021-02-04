import pandas as pd
import string
import argparse
import os

str2exact = {}

def main():
    print('\n#######################')
    print('Preprocess Part 1')
    print('#######################')

    data_dir = [(os.path.join(args.data_dir,'atomic2020.tsv'),'atomic2020'),
                (os.path.join(args.data_dir,'conceptnet.tsv'),'conceptnet'),
                (os.path.join(args.data_dir,'transomcs.tsv'),'transomcs'),
                (os.path.join(args.data_dir,'atomic.tsv'),'atomic')
    ]

    conceptnet_label_whitelist = {
        'AtLocation':None,#
        'CapableOf':None,#
        'Causes':None,#
        'CausesDesire':None,#
        'Desires':None,#
        'HasA':None,#
        'HasFirstSubevent':None,#
        'HasLastSubevent':None,#
        'HasPrerequisite':None,#
        'HasProperty':None,#
        'HasSubevent':None,#
        'MadeOf':None,#
        'MotivatedByGoal':None,#
        'NotDesires':None,#
        'PartOf':None,#
        'ReceivesAction':None,#
        'UsedFor':None,#
        'ObstructedBy':None
    }

    for f,kb in data_dir:
        print('\n{}: reading file'.format(kb))
        df_all = pd.read_csv(f, sep='\t')
        before_size = len(df_all)
        df_all.drop_duplicates(inplace=True)
        before_uniq = len(df_all)

        if kb.startswith('atomic'):
            df = df_all.copy()
        else:
            df = df_all[df_all['relation'].isin(conceptnet_label_whitelist)].copy()

        print('{}: processing head'.format(kb))
        df['head_exact'] = df[['head','relation']].apply(lambda x: str2exact[x['head']] if x['head'] in str2exact else clean_str(x['head'], kb, x['relation']), axis=1)

        print('{}: processing tail'.format(kb))
        df['tail_exact'] = df[['tail','relation']].apply(lambda x: str2exact[x['tail']] if x['tail'] in str2exact else clean_str(x['tail'], kb, x['relation']), axis=1)

        print('{}: writing processed file'.format(kb))

        d = '/'.join(f.split('/')[:-1]) + '/'
        df[['head', 'relation', 'tail', 'head_exact', 'tail_exact']].to_csv(d + kb + '_exact.tsv', index=False, sep='\t')



def clean_str(s_raw,kb, relation):
    if pd.isnull(s_raw):
        s_raw = ''

    s = s_raw.lower()
    if kb[:6] == 'atomic' and 'person' in s:
        s = s.replace('personx','person')
        s = s.replace('persony','person')
        s = s.replace('personz','person')

    s = s.strip().translate(str.maketrans('', '', string.punctuation))
    l = s.split()

    if not l:
        rv = ''
    elif kb[:6] == 'atomic' and (relation[0] in ["o","x"] or relation in ['isFilledBy', 'HinderedBy', 'isBefore', 'isAfter']) and l[0][:6]=='person':
        rv = ' '.join(l[1:])
    else:
        rv = ' '.join(l)

    str2exact[s_raw] = rv
    return rv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', help="directory with KBs in tab separated data files. Required headers and columns: [head, relation, tail]")

    args = parser.parse_args()

    main()
