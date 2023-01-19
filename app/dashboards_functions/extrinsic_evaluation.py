import pathlib
import sys
import keras
import pandas as pd
import numpy as np
from sklearn import metrics
from IPython.display import display
import ipywidgets as widgets

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from utils.geographical import *


def extrinsic_metrics(model_name):
    
    
    model = keras.models.load_model("models/"+model_name+".h5")
    
    
    aux_emb = pd.read_csv("embeddings/" + dict(pd.read_csv("Model#Emb.csv").values)[model_name] + ".csv") 

    tokenizer_df = aux_emb.loc[0:list(pd.isnull(aux_emb["sensor"])).index(True)-1,["sensor","id"]]
    tokenizer_df.id = [int(i) for i in tokenizer_df.id]
    tokenizer_df.index = tokenizer_df.id
    tokenizer_df = tokenizer_df[["sensor"]]

            
    df = pd.read_csv("trajectories/" + dict(pd.read_csv("Model#Data.csv").values)[model_name] + "_sample_trajs" +".csv") 

    df["0"] = df["0"].apply(lambda x : x.replace("[","").replace("]","").replace(",","").replace("\n","").replace("'",""))
    df["0"] = df["0"].apply(lambda x : x.split())
    df.rename(columns = {'Unnamed: 0':'trajectory_number'}, inplace = True)
    df["trajectory_number"] = df["trajectory_number"].apply(lambda x: str(x))


    Dict_Geo25 = { geo25.upper():vector  for geo25,vector in zip(list(tokenizer_df['sensor'].values),tokenizer_df.index)}
    df["0"] = df["0"].apply(lambda x:[ Dict_Geo25[geo25] for geo25 in x])
    validation = False
            

    x,y = padding_trajs(df["0"],model.input_shape[1] + 1)

    Df_Ext = pd.DataFrame(x)
    Df_Ext["y"] = y
    
    probs = model.predict(Df_Ext.values[:,:-1], verbose=0)
    

    y_ = np.argmax(model.predict(Df_Ext.values[:,:-1], verbose=0), axis=1)

    F1 = metrics.f1_score(Df_Ext["y"],y_, average='weighted')

    md = model_name[0:2]  
    if (md == 'LB'):
        range_labes = 371
    else:
        range_labes = 368

    AccTop1 = metrics.top_k_accuracy_score(Df_Ext["y"],probs, k=1, labels = np.array(range(0,range_labes)))
    AccTop3 = metrics.top_k_accuracy_score(Df_Ext["y"],probs, k=3, labels = np.array(range(0,range_labes)))
    AccTop5 = metrics.top_k_accuracy_score(Df_Ext["y"],probs, k=5, labels = np.array(range(0,range_labes)))

    return [model_name, AccTop1, AccTop3, AccTop5, F1]  

def show_accuracy(b, select_mutiples_models, select_mutiples_models_out):
            
    with select_mutiples_models_out:
        
        select_mutiples_models_out.clear_output()
        
        display(widgets.Label("Processing..."))
        
        w_sm_vals = select_mutiples_models.value
        list_out = []

        for i in range(len(w_sm_vals)):
            list_out.append(extrinsic_metrics(w_sm_vals[i]))

        df_out = pd.DataFrame(list_out, columns=['Model', 'Acc@1', 'Acc@3', 'Acc@5', 'F1-score'])
        df_out = df_out.style.highlight_max(axis=0, subset=['Acc@1','Acc@3','Acc@5', 'F1-score'], props='font-weight:bold;')
        df_out.format({'Acc@1': "{:.3}", 'Acc@3': "{:.3}", 'Acc@5': "{:.3}", 'F1-score': "{:.3}"})
        df_out.hide_index()
        
        select_mutiples_models_out.clear_output()
        
        display(df_out)