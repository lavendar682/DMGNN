# DriverMMGNN
## Requirements
- Python 3.10.9
- Pytorch 1.13.1+cu115
- Pytorch Geometric 2.4.0
- networkx 3.2
- numpy 1.23.5
- scipy 1.10.1
- scikit-learn 1.2.1
- pandas 1.5.3

## Graph&Feaature Construction
1, Get the adjacency list (.csv file) and node features (in data.rar)
2, Run the Graph2pyg

## Train the Model
1, Put graph and feature matrix in the same path as the main.py file
2, Change the value of parameter "name". You can rename the feature file and graph file as the PPI networks names.
3, Run the main.py
