import argparse
import subprocess
import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--credentials',
        action='store',
        help='credentials to authenticate with '
    )
    parser.add_argument(
        '--start_date',
        action='store',
        help='first date to fetch (Format: YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end_date',
        action='store',
        help='last date to fetch (Format: YYYY-MM-DD)'
    )
    parser.add_argument(
        '-p', '--project_id',
        action='store',
        help='project id, should be on of [dev-us, warehouse-prod]'
    )
    parser.add_argument(
        '--customer_table_id',
        action='store',
        help='customer table id to fetch data from'
    )
    parser.add_argument(
        '-o', '--output',
        action='store',
        help='directory where output is stored'
    )

    parser.add_argument(
        '-i', '--iana',
        action='store',
        help='iana ports list path'
    )

    parser.add_argument(
        '-w', '--false_positive_weight',
        type=float,
        action='store',
        default=0.95,
        help='false positive weight, in the range of (0,1), higher values result in fewer FPs (with the risk of fewer TPs as well)'
    )
    args = parser.parse_args()

    # Fetch the input data (train+test)
    print("Fetching data")
    # subprocess.call(f"python3 euler_fetching.py --credentials {args.credentials} --start_date {args.start_date} --end_date {args.end_date}"
    #                 f" --project_id {args.project_id} --customer_table_id {args.customer_table_id} --output {args.output}", shell=True)

    # Preprocess input data
    print("Preprocessing data")
    dates = args.start_date + '_' + args.end_date
    input_file_path = f'{args.output}/{args.customer_table_id}-{dates}.csv'
    print(input_file_path)
    last_training_day = (datetime.datetime.strptime(args.end_date, '%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    print(last_training_day)
    # subprocess.call(
    #     f"python3 euler_preprocessing.py --file {input_file_path} --date {last_training_day}"
    #     f" --iana {args.iana} --output {args.output}", shell=True)

    # Training
    # print("Training phase")
    training_file = f'{args.output}/adj_orig_dense_list.pkl'
    num_nodes_file = f'{args.output}/num_nodes.txt'
    # subprocess.call(
    #     f"python3 euler_training.py --file {training_file}"
    #     f" --num_nodes_file {num_nodes_file} --output {args.output} --predict", shell=True)

    # Evaluation
    print("Evaluation phase")
    test_csv = f'{args.output}/test_set_preprocessed.csv'
    model_path = f'{args.output}/gc_model.pkl'
    index_to_node = f'{args.output}/idx_to_node.pkl'
    subprocess.call(
        f"python3 euler_evaluation.py --model {model_path} --input_file {training_file} --csv_file {test_csv} --num_nodes_file {num_nodes_file}"
        f" --output {args.output} --index_to_node {index_to_node} --false_positive_weight {0.5}", shell=True)
