# AnomalousNeighbors


## About

In this project the ETP research team and Guardicore's Hunt team collaborates to introduce a ML solution to detect
anomalous neighbors from Hunt customers' data.

## Instructions:

1. clone the repo
2. initialize a virtual environment:  ``` python3 -m venv venv ```
3. Start the virtual environment: ``` . venv/bin/activate```
4. Install dependencies: ``` pip3 install -r requirements.txt```
5. Run full pipeline with: 

```python3 euler_full_pipeline.py --credentials {path_to_gcp_credentaials_json_file} --start_date {start_date --end_date {end_date (evaluation date)} --project_id {project_id, you probably want to pass warehous-prod here} --customer_table_id {table id of the customer within the project} --output {output directory, all files generated by the pipeline will be stored here} --iana raw_data/iana.csv ```

6. (Optionally) You can run each step by yourself if you want (basically, step 5 executes the whole pipeline: fetch->preprocess->train->evaluation), you can see what arguments are needed by running the python script with ``` -h ``` flag. For example: ``` python3 euler_fetching.py -h ```

Confirmed to work on MacOS 12.6.1, Python 3.9.12,  should work on any UNIX-like system. Windows should probably work as well, but I haven't tested it.


    
