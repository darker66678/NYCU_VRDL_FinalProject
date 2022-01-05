import json
import argparse
import pandas as pd


def main(args):
    f_src = open(args.json)
    src_json = json.load(f_src)
    src_csv = pd.read_csv(args.csv)
    ans_keys = src_csv['patientId']

    ans_dict = {
        'patientId': [key for key in ans_keys],
        'PredictionString': [''] * len(ans_keys)
    }

    ans_df = pd.DataFrame(ans_dict)

    for element in src_json:
        key = element['image_id']
        conf = element['score']
        x = int(element['bbox'][0] + 0.5)
        y = int(element['bbox'][1] + 0.5)
        w = int(element['bbox'][2] + 0.5)
        h = int(element['bbox'][3] + 0.5)
        row = ans_df.index[ans_df['patientId'] == key]
        ans_df['PredictionString'].loc[row] += '{} {} {} {} {} '.format(conf, x, y, w, h)

    target = str(args.json).split('/')[-1][:-5]
    target_fname = f'{target}.csv'
    ans_df.to_csv(target_fname, index=False)
    print('{} saved'.format(target_fname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default='answer.json', type=str, help='the source json file path, ex : answer.json')
    parser.add_argument('--csv', default='../data/stage_2_sample_submission.csv', type=str,
                        help='the sample cvs file path, ex : stage_2_sample_submission.csv')
    args = parser.parse_args()

    main(args)
