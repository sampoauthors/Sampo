import argparse
import pandas as pd
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+',
                        help='The path to the files containing extractions \
                        per review and item.')
    parser.add_argument('-n', '--number', type=int, default=10000,
                        help='The number of reviews to be considered.')
    args = parser.parse_args()
    return args


def measure_growth(files, number):
    frames, all_sets = [], [set() for _ in range(len(files))]
    for f in tqdm(files):
        dat = pd.read_csv(f)
        dat['ext'] = \
            dat.apply(lambda x: x['modifier'] + ';' + x['aspect'], axis=1)
        rev_data = dat.groupby('review_id').agg(ext=('ext', lambda x: set(x)))
        rev_data = rev_data.reset_index()
        frames.append(rev_data.sample(frac=1))
    print('reviews,' + ','.join(files))
    for i in range(number):
        to_print = ''
        for j, f in enumerate(frames):
            if i % 1000 == 0:
                to_print += ',' + str(len(all_sets[j]))
            all_sets[j] = all_sets[j].union(f.iloc[i]['ext'])
        if i % 1000 == 0:
            print(str(i) + ',' + to_print)


def main():
    args = parse_arguments()
    measure_growth(**vars(args))


if __name__ == "__main__":
    main()
