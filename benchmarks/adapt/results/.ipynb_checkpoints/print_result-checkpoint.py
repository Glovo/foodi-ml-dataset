'''
find ../logs/ -name *json -print0 | xargs -0 python print_result.py
'''
import sys
sys.path.append('../')

from retrieval.utils.file_utils import load_json
from collections import defaultdict

files = sys.argv[1:]

metrics = [
     'i2t_r1', 'i2t_r5', 'i2t_r10', 'i2t_meanr', 'i2t_medr',
     't2i_r1', 't2i_r5', 't2i_r10', 't2i_meanr', 't2i_medr',
]


def load_and_filter_file(file_path):

    result = load_json(file)
    print(result)
    result_filtered = {
        k.split('/')[1]: v
        for k, v in result.items()
        if k.split('/')[1] in metrics
    }


    res_line = '\t'.join(
            [f'{result_filtered[metric]:>4.2f}' for metric in metrics]
        )
    _file = '/'.join(file_path.split('/')[-3:])
    print(
        f'{_file:60s}\t{res_line}'
    )


def load_and_filter_files(file_path):

    result = load_json(file)
    result_filtered = defaultdict(dict)
    for k, v in result.items():
        try:
            data_name, metr = k.split('/')
        except:
            continue
        if metr not in metrics:
            continue
        result_filtered[data_name].update({metr: v})

    _file = '/'.join(file_path.split('/')[-3:-1])
    for data_name, vals in result_filtered.items():
        # print(data_name, [vals[m] for m in metrics])
        res_line = '\t'.join(
            [f'{vals[metric]:>4.2f}' for metric in metrics]
        )
        # print(_file, data_name, res_line)

        print(
            f'{_file:55s}\t{data_name:20s}\t{res_line}'
        )

for file in files:
    load_and_filter_file(file)
    load_and_filter_files(file)
