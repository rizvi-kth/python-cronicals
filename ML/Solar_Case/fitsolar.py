from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
os.getcwd()
# os.chdir("TorchServe/BERT_deploy")
# os.getcwd()

import torch

import readsolar as rs
solar_pdf = rs.read_solar()

# solar_pdf.isnull().sum()
print(f"Original Data shape: {solar_pdf.values.shape}")

# Filter by years
sol_2y_pdf = solar_pdf.loc["2017":"2018"]
print(f"Two Years Data shape: {sol_2y_pdf.values.shape}")

# Select columns
sols_pdf = sol_2y_pdf[['Power' , 'Month', 'Wind', 'Temp', 'Humid', 'GRad', 'DRad', 'WindD', 'Rain']]
sols_pdf.loc[sols_pdf.Power > 0].shape

sol_tp = torch.from_numpy(sols_pdf.values)

sol_tp.shape

