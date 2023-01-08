import pandas as pd
import pickle


def concat_fn(source_process_name:str, destination_port: int, delimiter: str = "!!!"):
    return source_process_name + delimiter + str(destination_port)


def process_iana(path):
    dtype = {
        'Service Name': str,
        'Port Number': str,
        'Transport Protocol': str,
        'Description': str,
        'Assignee': str,
        'Contact': str,
        'Registration Date': str,
        'Modification Date': str,
        'Reference': str,
        'Service Code': str,
        'Unauthorized Use Reported': str,
        'Assignment Notes': str}
    df_iana = pd.read_csv(path, header=0, dtype=dtype).fillna('N/A')
    df_iana = df_iana[~df_iana['Description'].str.contains('Unassigned')]  # ['Port Number']
    df_iana = df_iana[~df_iana['Description'].str.contains('N/A')]  # ['Port Number']
    l = list(df_iana[~df_iana['Description'].str.contains('Unassigned')]['Port Number'].unique())

    ll = []
    for s in l:
        try:
            if '-' in s:
                a, b = s.split('-')
                a, b = int(a), int(b)
                # create a range from the given values
                result = range(a, b + 1)
                # convert the range into list
                result = list(result)
                ll += result
            else:
                a = eval(s)
                ll.append(a)
        except Exception as e:
            print(s)

    return ll


def get_id_to_idx(df):
    print('updated get id')
    sources = list(df['source_node_id'].unique())
    destinations = list(df['destination_node_id'].unique())
    flows = list(df['flow_node_id'])
    unified = list(set(sources+destinations+flows))
    node_to_idx = {node:i for i,node in enumerate(unified)}
    idx_to_node = {i:node for node,i in node_to_idx.items()}
    return node_to_idx, idx_to_node


def save_node_to_idx_mapping(node_to_idx_path, idx_to_node_path, node_to_idx, idx_to_node):
    print(node_to_idx_path)
    with open(node_to_idx_path, 'wb') as handle:
        pickle.dump(node_to_idx, handle)

    with open(idx_to_node_path, 'wb') as handle:
        pickle.dump(idx_to_node, handle)


def load_node_to_idx_mapping(node_to_idx_path, idx_to_node_path):
    with open(node_to_idx_path, 'rb') as handle:
        node_to_idx = pickle.load(handle)

    with open(idx_to_node_path, 'rb') as handle:
        idx_to_node = pickle.load(handle)
    return node_to_idx, idx_to_node