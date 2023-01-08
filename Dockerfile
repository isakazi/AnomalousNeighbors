FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
RUN pip3 install pandas db-dtypes networkx google-cloud-bigquery pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
COPY . /home/AnomalousNeighbors
WORKDIR /home/AnomalousNeighbors
#RUN pip3 install -r requirements.txt
#ENTRYPOINT ["python3"]
#CMD ["euler/euler_training.py", "-f", "/home/AnomalousNeighbors/data/adj_orig_dense_list.pickle.backup", "-p"]
ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]

#RUN python -c "import torch; print(torch.__version__)"
