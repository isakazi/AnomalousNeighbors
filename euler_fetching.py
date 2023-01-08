import argparse
from datetime import datetime
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

projects = {"dev-us": "guardicore-68957042", "warehouse-prod":"guardicore-82669935"}

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
    args = parser.parse_args()
    start_date_aux = [int(s) for s in args.start_date.split('-')]
    start_date = datetime(year=start_date_aux[0], month=start_date_aux[1], day=start_date_aux[2]).date()
    end_date_aux = [int(s) for s in args.end_date.split('-')]
    end_date = datetime(year=end_date_aux[0], month=end_date_aux[1], day=end_date_aux[2]).date()
    project_id = projects[args.project_id]
    customer_table_id = args.customer_table_id
    query_string_specific_customer = f"""SELECT source_node_id, source_ip, source_username, source_process_name, destination_node_id, destination_ip, destination_port, 
    destination_domain, destination_process_name, connection_type, sample_datetime FROM `{project_id}.connections_us.{customer_table_id}`
    WHERE DATE(sample_datetime) <= '{end_date.isoformat()}'
    AND DATE(sample_datetime) >= '{start_date.isoformat()}'
    AND NOT(destination_node_id LIKE '%Unknown%' OR source_node_id LIKE '%Unknown%')
    """
    credentials = service_account.Credentials.from_service_account_file(args.credentials)
    bqclient = bigquery.Client(project=project_id, credentials=credentials)
    print('created credentials successfully')
    query_results = bqclient.query(query_string_specific_customer).result()
    print('got query results successfully')
    df = query_results.to_dataframe()
    print('converted to dataframe successfully')
    output_path = args.output
    dates = args.start_date + '_' + args.end_date
    df.to_csv(f'{output_path}/{customer_table_id}-{dates}.csv', encoding='utf-8', index=False)
