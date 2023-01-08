import argparse
import torch
import pickle
import pandas as pd

from utils.score_utils import get_score, get_optimal_cutoff
import utils.load_euler as vd
from utils import generators

def evaluate(model_path):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        action='store',
        help='pkl file of the trained model (output of euler_training)'
    )
    parser.add_argument(
        '-f', '--input_file',
        action='store',
        help='test data to evaluate model'
    )

    parser.add_argument(
        '-c', '--csv_file',
        action='store',
        help='csv file of the appropriate test data'
    )
    parser.add_argument(
        '-i', '--index_to_node',
        action='store',
        help='index to node name mapping (path to pkl file)'
    )
    parser.add_argument(
        '-w', '--false_positive_weight',
        type=float,
        action='store',
        default=0.9,
        help='false positive weight, in the range of (0,1), higher values result in fewer FPs (with the risk of fewer TPs as well)'
    )
    # parser.add_argument(
    #     '-i', '--index_to_node',
    #     action='store',
    #     help='node to index pkl file path'
    # )
    parser.add_argument(
        '-t',
        '--test_index',
        type=int,
        action='store',
        default=0,
        help='test data start index'
    )

    parser.add_argument(
        '-n', '--num_nodes_file',
        action='store',
        help='This loads the number of nodes in the training set (cannot be determined from the input because input is'
             'split by days and some nodes may appear in some days but not in others, so this needs to be provided '
             'explicitly)'
    )

    parser.add_argument(
        '-o', '--output',
        action='store',
        help='directory where output is stored'
    )
    args = parser.parse_args()
    with open(args.model, 'rb') as handle:
        gc_model = pickle.load(handle)
    df_test=pd.read_csv(args.csv_file, header=0)
    with open(args.num_nodes_file, 'r') as handle:
        num_nodes = int(handle.readline())
        print(f'num nodes is: {num_nodes}')
    data = vd.load_gc_data(args.input_file, num_nodes)

    with open(args.index_to_node, 'rb') as handle:
        idx_to_node = pickle.load(handle)
    with torch.no_grad():
        gc_model.eval()
        TEST_TS = args.test_index
        # print(f'all: {data.T}, end_tr:{data.T - TEST_TS}')
        print(f'all: {data.T}, end_tr:{TEST_TS}')
        # end_tr = data.T - TEST_TS
        end_tr = TEST_TS

        # zs = gc_model(data.x, data.eis, data.tr)[end_tr-1:]

        #     all test data, as well as last training timestamp from train (?)
        zs = gc_model(data.x, data.eis, data.all, data.all_attributes)[end_tr - 1:]
        print(f'{len(zs)}')
        p, n, z = generators.link_prediction(data, data.all, zs, start=end_tr - 1, include_tr=False)
        # p,n,z = generators.link_prediction(data, data.all, zs, start=0, end = end_tr-1, include_tr=False)
        t, f = gc_model.score_fn(p,n,z, as_probs=True)
        # t, f, tscores_backup, fscores_backup = custom_score_fn(gc_model, p, n, z, as_probs=True)
        dscores = get_score(t, f)
    my_optimal_cutoff = get_optimal_cutoff(t, f, fw=args.false_positive_weight)
    mask = t < my_optimal_cutoff
    suspected_entries = mask.nonzero().cpu().detach().numpy()
    indices = [l for i in suspected_entries.tolist() for l in i]
    sources = []
    destinations = []
    dest_ports = []
    scores = []
    print('after classification')
    for i in indices:
        id_1 = p[0][0][i].item()
        id_2 = p[0][1][i].item()
        sources.append(idx_to_node[id_1])
        destinations.append(idx_to_node[id_2])
        scores.append(t[i].item())
        # print(id_1, id_2, '\n', idx_to_node[id_1], idx_to_node[id_2])
    delimiter = '!!!'
    print(len(sources), len(destinations))
    dfs = []
    print('start')
    for source, dest in zip(sources, destinations):
        if delimiter in source and delimiter not in dest:
            s = source.split(delimiter)[0]
            p = source.split(delimiter)[1]
            my_df = df_test[(df_test['source_node_id'] == dest)
                            & (df_test['destination_port_mapped'] == int(p))
                            & (df_test['source_process_name'] == s)]
            if my_df.shape[0] == 0:
                my_df = df_test[(df_test['destination_node_id'] == dest)
                                & (df_test['destination_port_mapped'] == int(p))
                                & (df_test['source_process_name'] == s)]
        elif delimiter in dest and delimiter not in source:
            d = dest.split(delimiter)[0]
            p = dest.split(delimiter)[1]
            my_df = df_test[(df_test['source_node_id'] == source)
                            & (df_test['destination_port_mapped'] == int(p))
                            & (df_test['source_process_name'] == d)]
            if my_df.shape[0] == 0:
                my_df = df_test[(df_test['destination_node_id'] == source)
                                & (df_test['destination_port_mapped'] == int(p))
                                & (df_test['source_process_name'] == d)]
        else:
            print('should not happen ', source, ', ', dest)
        if my_df.shape[0] == 0:
            print('should not happen')
        else:
            dfs.append(my_df)
            my_df = None
    print('end')
    final_df = pd.concat(dfs)
    print(final_df.drop_duplicates().shape)
    final_df.drop_duplicates().to_csv(args.output+'/detections.csv', encoding='utf-8', index=False)








