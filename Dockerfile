FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
RUN pip3 install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

RUN python -c "import torch; print(torch.__version__)"
