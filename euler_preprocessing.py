import argparse
from preprocessing.preprocessing import Preprocessing
from preprocessing import preprocess_utils
import pickle

def get_ids_set(df):
    return set(list(df['source_node_id'].unique()) + list(df['destination_node_id'].unique()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file',
        action='store',
        help='Connections table path'
    )
    parser.add_argument(
        '-d', '--date',
        action='store',
        help='training last date (Format: YYYY-MM-DD)'
    )
    parser.add_argument(
        '-i', '--iana',
        action='store',
        help='iana ports list path'
    )
    parser.add_argument(
        '-o', '--output',
        action='store',
        help='directory where output is stored'
    )
    args = parser.parse_args()
    train_end_date = args.date
    df_init = Preprocessing.load_dataframe(args.file, args.iana)
    print('loaded dataframe successfully')

    df_init[df_init['sample_datetime']> train_end_date].to_csv(args.output+'/test_set_preprocessed.csv', encoding='utf-8', index=False)
    df = Preprocessing.preprocess_dataframe(df_init.fillna('NA'), True)
    print('preprocessed dataframe successfully')
    df_train = df[df['sample_datetime'] <= train_end_date]
    df_test = df[df['sample_datetime'] > train_end_date]

    df_train_ids = get_ids_set(df_train)
    df_test_ids = get_ids_set(df_test)

    diff = df_test_ids.difference(df_train_ids)

    df_only_known = df[(df['sample_datetime'] <= train_end_date)
                       | ((~df['source_node_id'].isin(diff)) & (~df['destination_node_id'].isin(diff)))]

    node_to_idx, idx_to_node = preprocess_utils.get_id_to_idx(df_only_known)
    df_only_known['source_node_id_idx'] = df_only_known['source_node_id'].apply(lambda s: node_to_idx[s])
    df_only_known['destination_node_id_idx'] = df_only_known['destination_node_id'].apply(lambda s: node_to_idx[s])
    df_only_known['flow_node_id_idx'] = df_only_known['flow_node_id'].apply(lambda s: node_to_idx[s])
    graphs = Preprocessing.get_graph_per_date(df_only_known, node_to_idx.values())
    matrices = Preprocessing.matrix_generation(graphs)
    output_file = args.output + '/' + 'adj_orig_dense_list.pkl'
    node_to_idx_output_file = args.output+ '/' + 'node_to_idx.pkl'
    idx_to_node_output_file = args.output + '/' + 'idx_to_node.pkl'
    num_nodes_file=args.output + '/' + 'num_nodes.txt'
    with open(num_nodes_file, 'w') as handle:
        handle.write(str(len(node_to_idx)))
    with open(output_file, 'wb') as handle:
        pickle.dump(matrices, handle)
    preprocess_utils.save_node_to_idx_mapping(node_to_idx_output_file, idx_to_node_output_file, node_to_idx, idx_to_node)