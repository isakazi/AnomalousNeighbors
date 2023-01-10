import argparse
import torch
import pickle
import pandas as pd

from utils.score_utils import get_score, get_optimal_cutoff
import utils.load_euler as vd
from utils import generators


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
        default=0.95,
        help='false positive weight, in the range of (0,1), higher values result in fewer FPs (with the risk of fewer TPs as well)'
    )
    parser.add_argument(
        '-t',
        '--test_size',
        type=int,
        action='store',
        default=1,
        help='test size, how many days from the end backwards are evaluated for (currently supporting evaluation for last day only, so do not change this)'
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
    assert args.test_size == 1
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
        TEST_TS = args.test_size
        end_tr = data.T - TEST_TS
        print(f'all: {data.T}, end_tr:{end_tr}')
        zs = gc_model(data.x, data.eis, data.all, data.all_attributes)[end_tr - 1:]
        p, n, z = generators.link_prediction(data, data.all, zs, start=end_tr - 1, include_tr=False)
        t, f = gc_model.score_fn(p,n,z, as_probs=True)
        dscores = get_score(t, f)
    my_optimal_cutoff = get_optimal_cutoff(t, f, fw=args.false_positive_weight)
    mask = t < my_optimal_cutoff
    suspected_entries = mask.nonzero().cpu().detach().numpy()
    indices = [l for i in suspected_entries.tolist() for l in i]
    print(len(indices))
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
    delimiter = '!!!'
    print('start')
    source_port_process_dest=[]
    print(len(sources), len(destinations))
    for source, dest in zip(sources, destinations):
        if delimiter in source:
            process, port = source.split(delimiter)
            port = int(port)
            asset = dest
        elif delimiter in dest:
            process, port = dest.split(delimiter)
            port = int(port)
            asset = source
        source_port_process_dest.append((asset, process, port))

    source_port_process_dest = list(set(source_port_process_dest))
    df_1 = df_test[
        df_test[["source_node_id", "source_process_name", "destination_port_mapped"]].apply(
            tuple, 1).isin(source_port_process_dest)]
    df_2 = df_test[
        df_test[["destination_node_id", "source_process_name", "destination_port_mapped"]].apply(
            tuple, 1).isin(source_port_process_dest)]
    final_df = pd.concat([df_1, df_2])
    final_df.drop_duplicates().to_csv(args.output+'/detections.csv', encoding='utf-8', index=False)





