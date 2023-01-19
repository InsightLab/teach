import pandas as pd
from IPython.display import display
import ipywidgets as widgets

import os
from os import listdir
import io
from io import StringIO

from copy import copy

from keras.models import load_model
import h5py
import pickle



def on_change_model_data_ml (change, model_data_multi_select):
    
    model_data_multi_select = change.new
    
def on_change_emb_data_ml (change, embedding_data_multi_select):

    embedding_data_multi_select = change.new

def on_change_data_name (change):

    teach_variable_states = {}

    with open('teach_variable_states.pickle', 'rb') as handle:
        teach_variable_states = pickle.load(handle)

    teach_variable_states["data_name"] = change.new

    with open('teach_variable_states.pickle', 'wb') as handle:
        pickle.dump(teach_variable_states, handle)


   
def on_change_model_name (change, model_name):

    model_name = change.new
    
def on_change_emb_name (change, embedding_name):

    embedding_name = change.new

def on_change_emb_ml (change, embedding_data_multi_select, embedding_data_multi_select_output, data_list, data_multi_select_dict_show, embedding_select): 
    
    embedding_select = change.new
    
    emb_data = pd.read_csv("Emb#Data.csv")
    
    if(embedding_select in list(emb_data["Emb"])):
        
        data_link_string = emb_data[emb_data["Emb"] == embedding_select]["Data"].values[0].split("%")
        aux_folder = listdir("data/")
        if('.csv' in aux_folder):
            aux_folder.remove('.csv')
        data_folder = [data.split(".")[0] for data in aux_folder]

        with embedding_data_multi_select_output:

            embedding_data_multi_select_output.clear_output()
            
            for data in data_folder:
                
                if(data in data_link_string):
                
                    data_multi_select_dict_show[data] = data + " " +"\u2714" 
                
                else:
                    
                    data_multi_select_dict_show[data] = data
            
            embedding_data_multi_select.options = [data_multi_select_dict_show[dl] for dl in data_list]
            display(embedding_data_multi_select)
        
    else:

        with embedding_data_multi_select_output:

            embedding_data_multi_select_output.clear_output()
            embedding_data_multi_select.options = [edl for edl in data_list]
            display(embedding_data_multi_select)

def on_change_model_ml(change, model_data_multi_select, model_data_multi_select_output, model_data_multi_select_dict_show, model_multi_select, data_list):
        
    model_multi_select = change.new
    
    model_data = pd.read_csv("Model#Data.csv")
    
    
    if(model_multi_select in list(model_data["Model"])):
        
        data_link_string = model_data[model_data["Model"] == model_multi_select]["Data"].values[0].split("%")
        aux_folder = listdir("data/")
        if('.csv' in aux_folder):
            aux_folder.remove('.csv')
        data_folder = [data.split(".")[0] for data in aux_folder]

        with model_data_multi_select_output:

            model_data_multi_select_output.clear_output()
            
            for data in data_folder:
                
                if(data in data_link_string):
                
                    model_data_multi_select_dict_show[data] = data + " " +"\u2714" 
                
                else:
                    
                    model_data_multi_select_dict_show[data] = data
            
            model_data_multi_select.options = [model_data_multi_select_dict_show[mdl] for mdl in data_list]
            display(model_data_multi_select)
        
    else:

        with model_data_multi_select_output:

            model_data_multi_select_output.clear_output()
            model_data_multi_select.options = [mdl for mdl in data_list]
            
            for mdl in data_list:
                [mdl] = mdl
            
            display(model_data_multi_select)


def upload_data(b, data_multi_select_dict_show, on_change_dataset, data_list, data_up, emb_data_multi_select, model_data_multi_select, model_data_multi_select_dict_show, out1, out2, out3, out4):

        
    with out1:
            
        
        out1.clear_output()
        
        display(widgets.Label("Processing..."))
        
        new_values = b.new
        keys = list(new_values.keys())

        for k in keys:
            data = StringIO(str(new_values[k]["content"],'utf-8'))
            df=pd.read_csv(data,index_col=0)
            df.to_csv("data/"+k.split(".")[0]+".csv")
            
        data_list = [data.split(".")[0] for data in listdir("data/")]
        
        if("" in data_list):
            data_list.remove("")
        
        data_up.options= data_list 
        data_up.value= [data_list[0]]
        
        out1.clear_output()
        
        display(data_up)
        
    with out2:

        out2.clear_output()
        
        keys = list(new_values.keys())
        
        for k in keys:
            data_multi_select_dict_show[k.split(".")[0]] = k.split(".")[0]

        
        emb_data_multi_select.options = [data_multi_select_dict_show[dl] for dl in data_list]
        display(emb_data_multi_select)
        
    with out3:

        out3.clear_output()
        
        keys = list(new_values.keys())
        
        for k in keys:
            model_data_multi_select_dict_show[k.split(".")[0]] = k.split(".")[0]

        
        model_data_multi_select.options = [model_data_multi_select_dict_show[mdl] for mdl in data_list]
        display(model_data_multi_select)
        
    with out4:

        out4.clear_output()
        text3 = widgets.Dropdown(description='',layout=widgets.Layout(width="80px"),options=data_list)
        text3.observe(on_change_dataset, names='value')
        display(text3)

def upload_embedding(b, model_choice, embedding_dictionary, embedding_dataframe, embedding_select, embedding_list_2, embedding_list_1, embedding_multi_select_up, out1, out2, out3):
    
    with out1:
            
        out1.clear_output()
        
        display(widgets.Label("Processing..."))
        
        new_values = b.new
        keys = list(new_values.keys())

        for k in keys:
            
            data = StringIO(str(new_values[k]["content"],'utf-8'))
            df=pd.read_csv(data,index_col=0)
            df.to_csv("embeddings/"+k.split(".")[0]+".csv")
            
        embedding_list_1 = [emb.split(".")[0] for emb in listdir("embeddings/")]
        
        if("" in embedding_list_1):
            embedding_list_1.remove("")
        
        pd.DataFrame({"Emb":embedding_list_1}).to_csv("Emb.csv",index=False)
        
        embedding_multi_select_up.options = embedding_list_1
        embedding_multi_select_up.value = [embedding_list_1[0]]
        
        out1.clear_output()
        
        display(embedding_multi_select_up)
        
    with out2:

        out2.clear_output()
        embedding_select = widgets.Select(options=embedding_list_1 + [''],description='Embeddings:',value='',layout=widgets.Layout(width="380px",height="180px"))
        embedding_select.observe(on_change_emb_ml, names = 'value')    
        display(embedding_select)   
    
        
    with out3:
        
        embedding_list_2 = list(pd.read_csv("Emb.csv").Emb)
        embedding_dataframe = pd.read_csv("Emb#Data.csv") 
        embedding_dictionary = { emb: embedding_dataframe[embedding_dataframe["Emb"]==emb].Data.values[0].split("%") for emb in list(embedding_dataframe.Emb)}

        model_choice.options = embedding_list_2 
        model_choice.value = embedding_list_2[0]

        out3.clear_output()
        display(model_choice)

def upload_model(b, model_multi_select_up, model_list, model_multi_select, out1, out2):
            
    with out1:
            
        
        out1.clear_output()
        
        display(widgets.Label("Processing..."))
        
        new_values = b.new
        keys = list(new_values.keys())
        
        Emb_list = list(pd.read_csv("Emb.csv")["Emb"])
        Model_Emb_df = pd.read_csv("Model#Emb.csv")

        for k in keys:
            data= io.BytesIO(new_values[k]["content"])
            model= h5py.File(data,'r')
            model_keras= load_model(model)
            model_keras.save("Models/"+k.split(".")[0]+".h5")
            
            for emb in Emb_list: # Sob modificação no futuro
            
                if(not(k.split(".")[0] in list(Model_Emb_df.Model) )):

                    if(k.split(".")[0] == emb.split("_")[0]):

                        Model_Emb_df = pd.DataFrame({"Model":list(Model_Emb_df["Model"]) + [k.split(".")[0]] , 
                                    "Emb":list(Model_Emb_df["Emb"]) + [emb]})
        
        
        Model_Emb_df.to_csv("Model#Emb.csv",index=False)
                    
            
        model_list = [model.split(".")[0] for model in listdir("models/")]
        
        if("" in model_list):
            model_list.remove("")
        
        pd.DataFrame({"Model":model_list}).to_csv("Model.csv",index=False)
        
        model_multi_select_up.options= model_list 
        model_multi_select_up.value = [model_list[0]]
        
        out1.clear_output()
        
        display(model_multi_select_up)
        
    with out2:

        out2.clear_output()
        model_multi_select = widgets.Select(options=model_list + [''],description='Models:',value='',layout=widgets.Layout(width="380px",height="180px"))
        model_multi_select.observe(on_change_model_ml, names = 'value') 
        display(model_multi_select)


def link_model(b, models_names, data_list,model_dictionary, model_dataframe, select_mutiples_models, model_multi_select, model_data_multi_select, model_data_multi_select_dict_show, out1, out2):

 
    model = model_multi_select.value

    data = [ mml.split(" ")[0] if("\u2714" in mml) else mml for mml in model_data_multi_select.value]
    model_data = pd.read_csv("Model#Data.csv")
    
    aux_folder = listdir("data/")
    if('.csv' in aux_folder):
        aux_folder.remove('.csv')
    data_folder = [d.split(".")[0] for d in aux_folder]
    
    if(model in list(model_data["Model"])):
    
        aux = []
        
        for d in data:
            if(not(d in model_data[model_data["Model"] == model]["Data"].values[0].split("%"))):
                aux.append(d)
                
        data = aux
        Index = model_data.loc[model_data["Model"] == model].index[0]
        model_data.loc[Index]["Data"] =  model_data[model_data["Model"] == model]["Data"][Index]+ "%"+ "%".join(data)
        model_data.to_csv("Model#Data.csv",index=False)
        
        data = model_data[model_data["Model"] == model]["Data"].values[0].split("%")

        
        with out1:
            
            out1.clear_output()
            
            for d in data_folder:
                
                if(d in data):
                    model_data_multi_select_dict_show[d] = d + " " +"\u2714" 
                    
                else:
                    model_data_multi_select_dict_show[d] = d 
            
            model_data_multi_select.options = [model_data_multi_select_dict_show[mdl] for mdl in data_list]
            display(model_data_multi_select)
        
    else:
        
    
        MD_df = pd.DataFrame({"Model":list(model_data["Model"]) + [model] , "Data":list(model_data["Data"]) + ["%".join(data)]})
        
        MD_df.to_csv("Model#Data.csv",index=False)
        
        with out1:
            
            out1.clear_output()
            
            for d in data_folder:
                
                if(d in data):
                    model_data_multi_select_dict_show[d] = d + " " +"\u2714" 
                    
                else:
                    model_data_multi_select_dict_show[d] = d  
            
            model_data_multi_select.options = [model_data_multi_select_dict_show[mdl] for mdl in data_list]
            display(model_data_multi_select)
    

        
    with out2:

        out2.clear_output()

        model_dataframe = pd.read_csv("Model#Data.csv")
        model_dictionary = {md:dt.split("%")[0] for md,dt in zip(model_dataframe.Model,model_dataframe.Data)}
        models_names = list(model_dataframe.Model)

        select_mutiples_models = widgets.SelectMultiple(options= models_names + [''], value=[''] , description='Models:', layout=widgets.Layout(width="420px",height="140px"), disabled=False) 
        display(select_mutiples_models)
                                
def link_embedding(b, model_choice, text_dataset, embedding_select, data_list, embedding_list, embedding_dataframe, embedding_dictionary, emb_data_multi_select, data_multi_select_dict_show, out1, out2, out3):
    
    emb = embedding_select.value
    data = [ eml.split(" ")[0] if("\u2714" in eml) else eml for eml in emb_data_multi_select.value]
    
    emb_data = pd.read_csv("Emb#Data.csv")
    
    aux_folder = listdir("data/")
    if('.csv' in aux_folder):
        aux_folder.remove('.csv')
    data_folder = [d.split(".")[0] for d in aux_folder]
    
    if(emb in list(emb_data["Emb"])):
    
        aux = []
        for d in data:
            if(not(d in emb_data[emb_data["Emb"] == emb]["Data"].values[0].split("%"))):
                aux.append(d)
                
        data = aux
        Index = emb_data.loc[emb_data["Emb"] == emb].index[0]
        emb_data.loc[Index]["Data"] =  emb_data[emb_data["Emb"] == emb]["Data"][Index]+ "%"+ "%".join(data)
        emb_data.to_csv("Emb#Data.csv",index=False)
        
        data = emb_data[emb_data["Emb"] == emb]["Data"].values[0].split("%")
        
        with out1:
            
            out1.clear_output()
            
            for d in data_folder:
                
                if(d in data):
                    data_multi_select_dict_show[d] = d + " " +"\u2714" 
                    
                else:
                    data_multi_select_dict_show[d] = d 
                
            emb_data_multi_select.options = [data_multi_select_dict_show[dl] for dl in data_list]
            display(emb_data_multi_select)
        
    else:
        
    
        EB_df = pd.DataFrame({"Emb":list(emb_data["Emb"]) + [emb] , "Data":list(emb_data["Data"]) + ["%".join(data)]})
        
        EB_df.to_csv("Emb#Data.csv",index=False)
        
        with out1:
            
            out1.clear_output()
            
            for d in data_folder:
                
                if(d in data):
                    data_multi_select_dict_show[d] = d + " " +"\u2714" 
                    
                else:
                    data_multi_select_dict_show[d] = d  
            
            emb_data_multi_select.options = [data_multi_select_dict_show[dl] for dl in data_list]
            display(emb_data_multi_select)
            
        
    with out2:
        
        embedding_list = list(pd.read_csv("Emb.csv").Emb)
        embedding_dataframe = pd.read_csv("Emb#Data.csv") 
        embedding_dictionary = { emb: embedding_dataframe[embedding_dataframe["Emb"]==emb].Data.values[0].split("%") for emb in list(embedding_dataframe.Emb)}
        model_choice.options = embedding_list 
        model_choice.value = embedding_list[0]
        out2.clear_output()
        display(model_choice)
        
    with out3:
        
        out3.clear_output()
        text_dataset.options =  [''] if(not(embedding_list[0] in embedding_dictionary.keys())) else embedding_dictionary[embedding_list[0]] + ['']
        text_dataset.value = ''
        display(text_dataset)
           

def unlink_model(b, models_names, select_mutiples_models, data_list,  model_data_multi_select, model_multi_select, model_dataframe, model_data_multi_select_dict_show, model_dictionary, out1, out2):
    
    model = model_multi_select.value
    data = [ mml.split(" ")[0] if("\u2714" in mml) else mml for mml in model_data_multi_select.value]
    
    model_data = pd.read_csv("Model#Data.csv")
    
    with_remove = copy(model_data[model_data["Model"] == model]["Data"].values[0].split("%")) 
    
    for d in data:
        
        with_remove.remove(d)
        
    if(len(with_remove)>0):
        
        Index = model_data.loc[model_data["Model"] == model].index[0]
        model_data.loc[Index]["Data"] =  "%".join(with_remove)
        model_data.to_csv("Model#Data.csv",index=False)
        
    else:
        
        model_data = model_data.drop(model_data[model_data.Model == model].index)
        model_data.to_csv("Model#Data.csv",index=False)
        
    with out1:

        out1.clear_output()

        for d in data:
            model_data_multi_select_dict_show[d] = d

        model_data_multi_select.options = [model_data_multi_select_dict_show[mdl] for mdl in data_list]
        display(model_data_multi_select)
        
        
    with out2:

        out2.clear_output()

        model_dataframe = pd.read_csv("Model#Data.csv")
        model_dictionary = {md:dt.split("%")[0] for md,dt in zip(model_dataframe.Model, model_dataframe.Data)}
        models_names = list(model_dataframe.Model)

        select_mutiples_models = widgets.SelectMultiple(options=models_names + [''], value=[''] , description='Models:', layout=widgets.Layout(width="420px",height="140px"), disabled=False) 

        display(select_mutiples_models)

def unlink_embedding(b, model_choice, text_dataset, embedding_select, emb_data_multi_select, data_list, embedding_dataframe, embedding_list, data_multi_select_dict_show, embedding_dictonary, out1, out2, out3):
        
    emb = embedding_select.value
    data = [ eml.split(" ")[0] if("\u2714" in eml) else eml for eml in emb_data_multi_select.value]
    
    emb_data = pd.read_csv("Emb#Data.csv")
    
    with_remove = copy(emb_data[emb_data["Emb"] == emb]["Data"].values[0].split("%")) 
    
    for d in data:
        
        with_remove.remove(d)
        
    if(len(with_remove)>0):
        
        Index = emb_data.loc[emb_data["Emb"] == emb].index[0]
        emb_data.loc[Index]["Data"] =  "%".join(with_remove)
        emb_data.to_csv("Emb#Data.csv",index=False)
        
    else:
        
        emb_data = emb_data.drop(emb_data[emb_data.Emb == emb].index)
        emb_data.to_csv("Emb#Data.csv",index=False)
        

    with out1:

        out1.clear_output()

        for d in data:
            data_multi_select_dict_show[d] = d

        emb_data_multi_select.options = [data_multi_select_dict_show[dl] for dl in data_list]
        display(emb_data_multi_select)
        
    with out2:
        
        Emb_list = list(pd.read_csv("Emb.csv").Emb)
        embedding_dataframe = pd.read_csv("Emb#Data.csv") 
        embedding_dictonary = { emb: embedding_dataframe[embedding_dataframe["Emb"]==emb].Data.values[0].split("%") for emb in list(embedding_dataframe.Emb)}
        model_choice.options = embedding_list 
        model_choice.value = embedding_list[0]
        out2.clear_output()
        display(model_choice)
        
    with out3:
        
        out3.clear_output()
        text_dataset.options =  [''] if(not(embedding_list[0] in embedding_dictonary.keys())) else embedding_dictonary[embedding_list[0]] + ['']
        text_dataset.value = ''
        
        display(text_dataset)


def remove_model(b, models_names, model_list, model_multi_select, select_mutiples_models, model_multi_select_up, model_dataframe, model_dictonary, out1, out2, out3):    
    
    model_data = pd.read_csv("Model#Data.csv")
    model_emb_df = pd.read_csv("Model#Emb.csv")
    models_df = pd.read_csv("Model.csv")
    
    mmlup_val = model_multi_select_up.value
    aux = []
    
    for ml in model_list:
        if(not(ml in mmlup_val)):
            aux.append(ml)
            
    model_list = aux
    
    with out1:
        
        out1.clear_output()
        
        for mv in mmlup_val: 
            
            os.remove("models/"+mv+".h5")
            models_df = models_df.drop(models_df[models_df.Model == mv].index)
            model_data = model_data.drop(model_data[model_data.Model == mv].index)
            model_emb_df = model_emb_df.drop(model_emb_df[model_emb_df.Model == mv].index)
            
        
        models_df.to_csv("Model.csv",index=False)
        model_data.to_csv("Model#Data.csv",index=False)
        model_emb_df.to_csv("Model#Emb.csv",index=False)
        
        if(len(model_list)>0):
                
            model_multi_select_up = widgets.SelectMultiple(options= model_list, description='Models:', value=[model_list[0]], layout=widgets.Layout(width="380px",height="180px") )
            
        else:
                
            model_multi_select_up = widgets.SelectMultiple(options= model_list,description='Models:', value=[], layout=widgets.Layout(width="380px",height="180px"))
        
        display(model_multi_select_up)
        
    with out2:

        out2.clear_output()
        model_multi_select.options = model_list + ['']
        model_multi_select.value = ''
        
        display(model_multi_select)
        
    with out3:

        out3.clear_output()

        model_dataframe = pd.read_csv("Model#Data.csv")
        model_dictonary = {md:dt.split("%")[0] for md,dt in zip(model_dataframe.Model, model_dataframe.Data)}
        models_names = list(model_dataframe.Model)

        select_mutiples_models = widgets.SelectMultiple( options= models_names + [''], value=[''] , description='Models:', layout=widgets.Layout(width="420px",height="140px"), disabled=False) 

        display(select_mutiples_models)
            
def remove_data(b, model_choice, text_dataset, on_change_dataset, on_change_model_data_ml, select_mutiples_models, data_list, data_up, models_names, embedding_list, model_dataframe, embedding_dataframe, model_dictionary, data_multi_select_dict_show, 
                    model_data_multi_select_dict_show, embedding_dictionary, out1, out2, out3, out4, out5, out6, out7):

    emb_data = pd.read_csv("Emb#Data.csv")
    
    dup_val = data_up.value
    aux = []
    
    for d in data_list:
        if(not(d in dup_val)):
            aux.append(d)
        
    data_list = aux
    
    with out1:
        
        out1.clear_output()
        
        for d in dup_val: 
            
            os.remove("data/"+d+".csv")
                
            for ID in emb_data.index:
                    
                L_data = emb_data.loc[ID].Data.split("%")
                
                if(d in L_data):
                    
                    L_data.remove(d)
                    
                    if(len(L_data)==0):
                        
                        emb_data = emb_data.drop(emb_data[emb_data.Emb == emb_data.loc[ID].Emb].index)
                        
                    else:
                        
                        emb_data.loc[ID,"Data"] = "%".join(L_data)
            
        emb_data.to_csv("Emb#Data.csv",index=False)
                    
        
        if(len(data_list)>0):

            data_up = widgets.SelectMultiple(options=data_list,description='Data:', value=[data_list[0]],layout=widgets.Layout(width="380px",height="180px"))
        else:
    
                data_up = widgets.SelectMultiple(options=data_list,description='Data:', value=[],layout=widgets.Layout(width="380px",height="180px"))
        
        display(data_up)
        
    with out2:

        out2.clear_output()
        
        for ex in dup_val:
            data_multi_select_dict_show.pop(ex)
            
        emb_data_ml = widgets.SelectMultiple(options=[data_multi_select_dict_show[dl] for dl in data_list],description='Data:', layout=widgets.Layout(width="380px",height="180px"))
        emb_data_ml.observe(on_change_emb_data_ml, names = 'value')
        emb_data_ml.value = []
        display(emb_data_ml)
        
    with out3:

        out3.clear_output()
        
        for ex in dup_val:
            model_data_multi_select_dict_show.pop(ex)
        
        model_data_ml = widgets.SelectMultiple(options=[model_data_multi_select_dict_show[mdl] for mdl in data_list],description='Data:', layout=widgets.Layout(width="380px",height="180px"))
        model_data_ml.observe(on_change_model_data_ml, names = 'value')
        model_data_ml.value = []
        display(model_data_ml)
        
    with out4:

        out4.clear_output()

        model_dataframe = pd.read_csv("Model#Data.csv")
        model_dictionary = {md:dt.split("%")[0] for md,dt in zip(model_dataframe.Model, model_dataframe.Data)}
        models_names = list(model_dataframe.Model)

        select_mutiples_models = widgets.SelectMultiple( options=models_names + [''], value=[''] , description='Models:', layout=widgets.Layout(width="420px",height="140px"), disabled=False) 

        display(select_mutiples_models)
        
        
    with out5:

        out5.clear_output()
        text3 = widgets.Dropdown(description='',layout=widgets.Layout(width="80px"),options=data_list)
        text3.observe(on_change_dataset, names='value')
        display(text3)
        
    with out6:
        
        embedding_list = list(pd.read_csv("Emb.csv").Emb)
        embedding_dataframe = pd.read_csv("Emb#Data.csv") 
        embedding_dictionary = { emb: embedding_dataframe[embedding_dataframe["Emb"]==emb].Data.values[0].split("%") for emb in list(embedding_dataframe.Emb)}
        model_choice.options = embedding_list 
        model_choice.value = embedding_list[0]
        out6.clear_output()
        display(model_choice)
        
    with out7:
        
        out7.clear_output()        
        text_dataset.options =  [''] if(not(embedding_list[0] in embedding_dictionary.keys())) else embedding_dictionary[embedding_list[0]] + ['']
        text_dataset.value = ''
        display(text_dataset)
        
def remove_embedding(b, model_choice, embedding_list_2, embedding_select, embedding_multi_select_up, embedding_list_1, embedding_dataframe, embedding_dictionary, out1, out2, out3):
    
    emb_data = pd.read_csv("Emb#Data.csv")
    emb_df = pd.read_csv("Emb.csv")
    model_emb_df = pd.read_csv("Model#Emb.csv")
    
    emlup_val = embedding_multi_select_up.value
    aux = []

    for emb in embedding_list_2:
        if(not(emb in emlup_val)):
            aux.append(emb)
            
    
    embedding_list_2 = aux
    
    with out1:
        out1.clear_output()
        
        for emb in emlup_val: 
            
            os.remove("embeddings/"+emb+".csv")
            emb_df = emb_df.drop(emb_df[emb_df.Emb == emb].index)
            emb_data = emb_data.drop(emb_data[emb_data.Emb == emb].index)
            model_emb_df = model_emb_df.drop(model_emb_df[model_emb_df.Emb == emb].index)
            
        
        emb_df.to_csv("Emb.csv",index=False)
        emb_data.to_csv("Emb#Data.csv",index=False)
        model_emb_df.to_csv("Model#Emb.csv",index=False)
        
        
        
        if(len(embedding_list_2)>0):
            embedding_multi_select_up = widgets.SelectMultiple(options= embedding_list_2,description='Emb:', value=[embedding_list_2[0]],layout=widgets.Layout(width="380px",height="180px"))
            
        else:
            embedding_multi_select_up = widgets.SelectMultiple(options= embedding_list_2,description='Emb:', value=[],layout=widgets.Layout(width="380px",height="180px"))
        
        display(embedding_multi_select_up)
        
    with out2:

        out2.clear_output()
        embedding_select.options = embedding_list_2 + ['']
        embedding_select.value = ''
        display(embedding_select)
        
        
    with out3:
        
        embedding_list_1 = list(pd.read_csv("Emb.csv").Emb)
        embedding_dataframe = pd.read_csv("Emb#Data.csv") 
        embedding_dictionary = { emb: embedding_dataframe[embedding_dataframe["Emb"]==emb].Data.values[0].split("%") for emb in list(embedding_dataframe.Emb)}
        model_choice.options = embedding_list_1 
        model_choice.value = embedding_list_1[0]
        out3.clear_output()
        display(model_choice)


def rename_model(b, model_name, models_names, model_list, model_multi_select, model_multi_select_up, model_dataframe, model_dictionary, select_mutiples_models, out1 , out2 , out3 ):
        
    model_data = pd.read_csv("Model#Data.csv")
    models_df = pd.read_csv("Model.csv")
    model_emb_df = pd.read_csv("Model#Emb.csv")
    
    mmlup_val = model_multi_select_up.value
    aux = []
    
    
    for ml in model_list:#
        if(ml in mmlup_val):
            aux.append(model_name)
        else:
            aux.append(ml)
            
    model_list = aux
    
    with out1:
        
        out1.clear_output()
        
        for mv in mmlup_val: 
            
            os.rename("models/"+mv+".h5", "models/"+model_name+".h5")
            models_df.replace(mv,model_name,inplace=True)
            model_data.replace(mv,model_name,inplace=True)
            model_emb_df.replace(mv,model_name,inplace=True)
            
        
        models_df.to_csv("Model.csv",index=False)
        model_data.to_csv("Model#Data.csv",index=False)
        model_emb_df.to_csv("Model#Emb.csv",index=False)
                

        if(len(model_list)>0):

            model_multi_select_up = widgets.SelectMultiple(options=model_list,description='Models:', value=[model_list[0]],layout=widgets.Layout(width="380px",height="180px"))
        else:
    
            model_multi_select_up = widgets.SelectMultiple(options=model_list,description='Models:', value=[],layout=widgets.Layout(width="380px",height="180px"))
        
        display(model_multi_select_up)
        
    with out2:

        out2.clear_output()
        model_multi_select = widgets.Select(options=model_list + [''],description='Models:',value='',layout=widgets.Layout(width="380px",height="180px"))
        model_multi_select.observe(on_change_model_ml, names = 'value')
        model_multi_select.value = model_list[0]
        display(model_multi_select)
        
    with out3:

        out3.clear_output()

        model_dataframe = pd.read_csv("Model#Data.csv")
        model_dictionary = {md:dt.split("%")[0] for md,dt in zip(model_dataframe.Model, model_dataframe.Data)}
        models_names = list(model_dataframe.Model)

        select_mutiples_models = widgets.SelectMultiple( options=models_names + [''], value=[''] , description='Models:', layout=widgets.Layout(width="420px",height="140px"), disabled=False) 

        display(select_mutiples_models)
      
def rename_data(b, model_choice, text_dataset, on_change_dataset, on_change_model_data_ml, on_change_emb_data_ml, models_names, data_list, data_up, select_mutiples_models, model_data_multi_select_dict_show, data_multi_select_dict_show, model_dataframe, model_dictionary, 
                    embedding_list, embedding_dataframe, embedding_dictionary, out1, out2, out3, out4, out5, out6, out7):

    teach_variable_states = {}
    
    with open('teach_variable_states.pickle', 'rb') as handle:
        teach_variable_states = pickle.load(handle)
    
    data_name =  teach_variable_states["data_name"]

    emb_data = pd.read_csv("Emb#Data.csv")
    
    print(teach_variable_states["data_list"])
    dup_val = data_up.value
    aux = []
    
    for d in data_list:
        if(d in dup_val):
            aux.append(data_name)
        else:
            aux.append(d)
            
    data_list = aux
    
    teach_variable_states["data_list"] = data_list

    with open('teach_variable_states.pickle', 'wb') as handle:
        pickle.dump(teach_variable_states, handle)
    
    with out1:
        
        out1.clear_output()
        
        for d in dup_val:
            
            os.rename("data/"+d+".csv","data/"+data_name+".csv")
        
            for ID in emb_data.index:

                L_data = emb_data.loc[ID].Data.split("%")

                aux_ld =[]

                for ld in L_data:

                    if(ld==d):
                        aux_ld.append(data_name)
                    else:
                        aux_ld.append(ld)

                L_data = aux_ld                   

                emb_data.loc[ID,"Data"] = "%".join(L_data)


        emb_data.to_csv("Emb#Data.csv",index=False)
            
        
        if(len(data_list)>0):

            data_upload = widgets.SelectMultiple(options=data_list,description='Data:', value=[data_list[0]],layout=widgets.Layout(width="380px",height="180px"))
        else:
    
            data_upload = widgets.SelectMultiple(options=data_list,description='Data:', value=[],layout=widgets.Layout(width="380px",height="180px"))
        
        display(data_upload)
        
    with out2:

        out2.clear_output()
        
        for ex in dup_val:
            
            if("\u2714" in data_multi_select_dict_show[ex]):
                
                data_multi_select_dict_show[data_name] = data_name  + " " +"\u2714" 
                
            else:
                
                data_multi_select_dict_show[data_name] = data_name
                
            data_multi_select_dict_show.pop(ex)

        emb_data_ml = widgets.SelectMultiple(options=[data_multi_select_dict_show[edl] for edl in data_list],description='Data:', layout=widgets.Layout(width="380px",height="180px"))
        emb_data_ml.observe(on_change_emb_data_ml, names = 'value')
        display(emb_data_ml) 
        
    with out3:

        out3.clear_output()
        
        for ex in dup_val:
            
            if("\u2714" in model_data_multi_select_dict_show[ex]):
                
                model_data_multi_select_dict_show[data_name] = data_name + " " +"\u2714" 
                
            else:
                
                model_data_multi_select_dict_show[data_name] = data_name
                
            model_data_multi_select_dict_show.pop(ex)

        model_data_ml = widgets.SelectMultiple(options=[model_data_multi_select_dict_show[mdl] for mdl in data_list],description='Data:', layout=widgets.Layout(width="380px",height="180px"))
        model_data_ml.observe(on_change_model_data_ml, names = 'value')
        display(model_data_ml)
        
    with out4:

        out4.clear_output()

        model_dataframe = pd.read_csv("Model#Data.csv")
        model_dictionary = {md:dt.split("%")[0] for md,dt in zip(model_dataframe.Model,model_dataframe.Data)}
        models_names = list(model_dataframe.Model)

        select_mutiples_models = widgets.SelectMultiple( options=models_names + [''], value=[''] , description='Models:', layout=widgets.Layout(width="420px",height="140px"), disabled=False) 

        display(select_mutiples_models)
        
        
    with out5:

        out5.clear_output()
        text3 = widgets.Dropdown(description='',layout=widgets.Layout(width="80px"),options=data_list)
        text3.observe(on_change_dataset, names='value')
        display(text3)
        
    with out6:
        
        embedding_list = list(pd.read_csv("Emb.csv").Emb)
        embedding_dataframe = pd.read_csv("Emb#Data.csv") 
        embedding_dictionary = { emb: embedding_dataframe[embedding_dataframe["Emb"]==emb].Data.values[0].split("%") for emb in list(embedding_dataframe.Emb)}
        model_choice.options = embedding_list 
        model_choice.value = embedding_list[0]
        out6.clear_output()
        display(model_choice)
        
    with out7:
        
        out7.clear_output()
        text_dataset.options =  [''] if(not(embedding_list[0] in embedding_dictionary.keys())) else embedding_dictionary[embedding_list[0]] + ['']
        text_dataset.value = ''
        display(text_dataset)
        
def rename_embedding(b, embedding_name, model_choice, embedding_list_1, embedding_select, embedding_multi_select_up, on_change_emb_ml, embedding_list_2, embedding_dataframe, embedding_dictionary, out1, out2, out3):    
                
    
    emb_data = pd.read_csv("Emb#Data.csv")
    emb_df = pd.read_csv("Emb.csv")
    model_emb_df = pd.read_csv("Model#Emb.csv")
    
    emlup_val = embedding_multi_select_up.value
    aux = []
    
    
    for emb in embedding_list_2,:

        if(emb in emlup_val):
            aux.append(embedding_name)
        else:
            aux.append(emb)
            
    embedding_list_2, = aux
    
    with out1:
        
        out1.clear_output()
        
        for emb in emlup_val: 
            
            os.rename("embeddings/"+emb+".csv", "embeddings/"+embedding_name+".csv")
            emb_df.replace(emb,embedding_name,inplace=True)
            emb_data.replace(emb,embedding_name,inplace=True)
            model_emb_df.replace(emb,embedding_name,inplace=True)
            
        
        emb_df.to_csv("Emb.csv",index=False)
        emb_data.to_csv("Emb#Data.csv",index=False)
        model_emb_df.to_csv("Model#Emb.csv",index=False)
        
        if(len(embedding_list_2,)>0):

            embedding_multi_select_up, = widgets.SelectMultiple(options= embedding_list_2,description='Emb:', value=[embedding_list_2,[0]],layout=widgets.Layout(width="380px",height="180px"))
        else:
    
            embedding_multi_select_up, = widgets.SelectMultiple(options= embedding_list_2, description='Emb:', value=[],layout=widgets.Layout(width="380px",height="180px"))
                
        display(embedding_multi_select_up,)
        
    with out2:

        out2.clear_output()
        embedding_select = widgets.Select(options= embedding_list_2 + [''],description='Embeddings:',value='',layout=widgets.Layout(width="380px",height="180px"))
        embedding_select.observe(on_change_emb_ml, names = 'value')
        embedding_select.value = embedding_list_2[0]
        display(embedding_select)
        
        
    with out3:
        
        embedding_list_1 = list(pd.read_csv("Emb.csv").Emb)
        embedding_dataframe = pd.read_csv("Emb#Data.csv") 
        embedding_dictionary = { emb: embedding_dataframe[embedding_dataframe["Emb"]==emb].Data.values[0].split("%") for emb in list(embedding_dataframe.Emb)}
        model_choice.options = embedding_list_1
        model_choice.value = embedding_list_1[0]
        out3.clear_output()
        display(model_choice)


