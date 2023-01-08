import pandas as pd
import numpy as np
import preprocessing.preprocess_utils as preprocess_utils
import networkx as nx
from torch_geometric.utils.convert import from_scipy_sparse_matrix

class Preprocessing(object):
    def __init__(self):
        pass

    @staticmethod
    def load_dataframe(df_path, iana_path):
        dtypes = {
            'source_node_id': 'string',
            # 'source_node_type': 'string',
            'source_ip': 'string',
            'source_username': 'string',
            # 'source_process': 'string',
            # 'source_process_id': 'string',
            'source_process_name': 'string',
            # 'source_process_full_path': 'string',
            'destination_node_id': 'string',
            # 'destination_node_type': 'string',
            'destination_ip': 'string',
            'destination_port': int,
            'destination_domain': 'string',
            # 'destination_process': 'string',
            # 'destination_process_id': 'string',
            'destination_process_name': 'string',
            # 'destination_process_full_path': 'string',
            'connection_type': 'string',
            # 'flow_id': 'string',
            # 'ip_protocol': 'string',
            'sample_datetime': 'string'
        }
        df = pd.read_csv(df_path, header=0, dtype=dtypes)
        print("done loading")
        iana_list = set(preprocess_utils.process_iana(iana_path))
        known_ports = set(list(range(0, 1025)))
        iana_list = iana_list.union(known_ports)
        df['destination_port_mapped'] = df['destination_port'].apply(lambda p: p if p in iana_list else -1)
        print('done port mapping')
        # df_only_known = Preprocessing.preprocess_dataframe(df.fillna('NA'), True)
        return df

    @staticmethod
    def preprocess_dataframe(df, add_weight=False):
        df = df[['sample_datetime', 'source_node_id', 'source_process_name', 'destination_port_mapped',
                 'destination_node_id', 'connection_type']]
        if add_weight:
            df = df.groupby(['sample_datetime', 'source_node_id', 'source_process_name', 'destination_port_mapped',
                             'destination_node_id'], as_index='False').agg(
                {'connection_type': 'count'}).reset_index()
        df['flow_node_id'] = df.apply(lambda row: preprocess_utils.concat_fn(row.source_process_name,
                                                                             row.destination_port_mapped),
                                      axis=1)
        return df

    @staticmethod
    def get_graph_with_attributes(df, source='source_node_id', target='destination_node_id', node_to_idx=None,
                                  idx_to_node=None, process_to_idx=None, idx_to_process=None):
        G = nx.from_pandas_edgelist(df,
                                    source='source_node_id_idx',
                                    target='flow_node_id_idx',
                                    edge_attr=['connection_type'],
                                    # edge_attr = ['source_process_name_idx', 'destination_port', 'connection_type_int'],
                                    create_using=nx.DiGraph)
        H = nx.from_pandas_edgelist(df,
                                    source='flow_node_id_idx',
                                    target='destination_node_id_idx',
                                    edge_attr=['connection_type'],
                                    # edge_attr = ['source_process_name_idx', 'destination_port', 'connection_type_int'],
                                    create_using=nx.DiGraph)

        return nx.compose(G, H)

    @staticmethod
    def get_graph_per_date(df, nodes):
        sample_dates = sorted(list(df['sample_datetime'].unique()))
        print(sample_dates)
        graphs = []

        # # source_node_id_idx destination_node_id_idx source_process_name_idx
        for date in sample_dates:
            h = Preprocessing.get_graph_with_attributes(df[df['sample_datetime'] == date])
            g = nx.DiGraph((nx.induced_subgraph(h, nodes)))
            graphs.append(g)
            for node in nodes:
                g.add_node(node)
        return graphs

    @staticmethod
    def matrix_generation(graphs):
        matrices = []
        for graph in graphs:
            graph_undirected = graph.to_undirected()
            self_loop = list(nx.selfloop_edges(graph_undirected))
            graph_undirected.remove_edges_from(self_loop)
            matrix = nx.to_scipy_sparse_matrix(graph_undirected,
                                               dtype=np.int64,
                                               nodelist=sorted(graph_undirected.nodes()),
                                               weight='connection_type',
                                               format='coo')
            matrices.append(from_scipy_sparse_matrix(matrix))
        return matrices






