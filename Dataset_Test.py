import Dataset 
import numpy as np 
import pyvis 
import sys 
dataset = Dataset.Dataset()
index = 0 

while True :
    if index % 100 == 0 : 
        print(f"{index}/{len(dataset.cl_info_df)}\b") 
    info = dataset.cl_info_df.loc[index, :]
    index += 1
    checklist_df = dataset.cl_df.loc[dataset.cl_df["proc_id"] == info.proc_id]
    try : 
        input_tensor, subgraph = dataset.CreateInputTensor(info.title_perso, info.subtitle, info.text_fr)
        output_tensor = dataset.Checklist2Matrix(checklist_df, subgraph) 
    except Exception as e: 
        print(e)  
        pass
    if not (input_tensor is None): 
        with np.printoptions(threshold=np.inf) : 
            print(info)
            print(checklist_df) 
            print(input_tensor.shape) 
            # print(input_tensor[:150-24, :])
            # print("\n\n") 
            # print(input_tensor[150-24:, :]) 
            print("\n\n") 
            print(output_tensor) 
            net = pyvis.network.Network("700px", "700px") 
            net.from_nx(subgraph) 
            net.show("net.html") 
            break
