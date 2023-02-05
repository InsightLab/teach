import pandas as pd
from IPython.display import display, HTML
import ipywidgets as widgets

import os
from os import listdir
import io
from io import StringIO

from copy import copy

from keras.models import load_model
import h5py

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dashboards_classes.extrinsic_evaluation import *
from dashboards_classes.view_trajectory_data import *
from dashboards_classes.intrinsic_evaluation import *
from utils.output import *



class Teach_Main:
    
    def __init__(self, intrinsic_evaluation_output):

        self.data_upload = ""
        
        self.model_multi_select = ""
        self.model_select= ""
        self.model_multi_select_upload = ""
        self.model_data_select_multiple= ""
        self.model_data_multi_select = ""

        self.embedding_multi_select = ""
        self.embedding_select = ""
        self.embedding_multi_select_upload = ""
        self.embedding_data_select_multiple = ""
        self.embedding_data_multi_select = ""

        self.data_upload_output = ""

        self.model_multi_select_upload_output = ""
        self.model_data_select_multiple_output = ""
        self.model_select_output = ""

        self.embedding_multi_select_upload_output = ""
        self.embedding_select_output = ""
        self.embedding_data_select_multiple_output = ""

        self.model_name = ""
        self.data_name = ""
        self.embedding_name = ""

        self.intrinsic_evaluation_output = intrinsic_evaluation_output

        # The classes associated with other screens , composing the Teach_Main class as an attribute
        self.VTDO = View_Trajectory_Data_Output()
        self.EEO = Extrinsic_Evaluation_Output()
        self.IEO = Intrinsic_Evaluation_Output(self.intrinsic_evaluation_output)

        self.Model_df = pd.read_csv("Model#Data.csv")
        self.Model_dict = {md:dt.split("%")[0] for md,dt in zip(self.Model_df.Model, self.Model_df.Data)}
        self.models_names = list(self.Model_df.Model)

        self.data_list  = [dt.split(".")[0] for dt in listdir("data/")]
        self.model_list = [model.split(".")[0] for model in listdir("models/")]
        self.embedding_list   = [eb.split(".")[0] for eb in listdir("embeddings/")]
    
        
        if("" in self.data_list):
            self.data_list.remove("")  
            
        if("" in self.data_list):
            self.data_list.remove("")
            
        if("" in self.embedding_list):
            self.embedding_list.remove("")

        self.spc1 = widgets.Label("",layout=widgets.Layout(width="9%"))
        self.spc2 = widgets.Label("",layout=widgets.Layout(width="6%"))

        self.model_data_multi_select_dict_show = {mdl:mdl for mdl in self.data_list}
        self.data_multi_select_dict_show = {dl:dl for dl in self.data_list}



        # If the data_list is empty value element is empty
        if(len(self.data_list)>0):
        
            self.data_upload = widgets.SelectMultiple(options=self.data_list,description='Data:', value=[self.data_list[0]],layout=widgets.Layout(width="380px",height="180px"))
        else:
            
            self.data_upload = widgets.SelectMultiple(options=self.data_list,description='Data:', value=[],layout=widgets.Layout(width="380px",height="180px"))
        
        # Screen that shows the datasets of type lat_lon that are uploaded to the tool
        self.data_upload_output = out(self.data_upload)

        # The space to put the name of the lat_lon dataset to be renamed.
        self.data_name_label = widgets.Text(description='',layout=widgets.widgets.Layout(width="250px"))
        self.data_name_label.observe(self.on_change_data_name, names = 'value')

        # Data upload button
        self.data_upload_button = widgets.FileUpload(acept=".csv",multiple=True,layout=widgets.Layout(width="150px"))
        self.data_upload_button.observe(self.upload_data, names = "value")

        # Data rename button
        self.data_rename_button = widgets.Button(description="Rename", layout=widgets.Layout(width="90px"))
        self.data_rename_button.style.button_color = "lightgray"
        self.data_rename_button.on_click(self.rename_data)

        # Data remove button
        self.data_remove_button = widgets.Button(description="Remove", layout=widgets.Layout(width="150px"))
        self.data_remove_button.on_click(self.remove_data)

        self.data_upload_box = widgets.HBox(children=[widgets.VBox([self.data_upload_button, self.data_remove_button, self.spc1]), self.spc1,widgets.VBox([self.data_upload_output,self.spc2, 
                                                                            widgets.HBox([self.spc1,self.spc1,self.spc2, self.data_name_label, self.data_rename_button])])])




        # If model_list is empty value element is empty
        if(len(self.model_list)>0):
        
            self.model_multi_select_upload = widgets.SelectMultiple(options=self.model_list,description='Models:', value=[self.model_list[0]],layout=widgets.Layout(width="380px",height="180px"))

        else:
            
            self.model_multi_select_upload = widgets.SelectMultiple(options=self.model_list,description='Models:', value=[],layout=widgets.Layout(width="380px",height="180px"))
        
        # the screen that shows the models that are uploaded to the tool
        self.model_multi_select_upload_output = out(self.model_multi_select_upload)

        # Model upload button
        self.model_upload_button = widgets.FileUpload(acept=".h5",multiple=True,layout=widgets.Layout(width="150px"))
        self.model_upload_button.observe(self.upload_model, names="value")

        # Model remove button
        self.model_remove_button = widgets.Button(description="Remove", layout=widgets.Layout(width="150px"))
        self.model_remove_button.on_click(self.remove_model)
        
        # Model raname button
        self.model_rename_button = widgets.Button(description="Rename", layout=widgets.Layout(width="90px"))
        self.model_rename_button.style.button_color = "lightgray"
        self.model_rename_button.on_click(self.rename_model)
        
        # The space to put the name of model to be renamed.
        self.model_name_label = widgets.Text(description='',layout=widgets.widgets.Layout(width="250px"))
        self.model_name_label.observe(self.on_change_model_name, names = 'value')
            
        
        self.model_upload_box = widgets.HBox(children=[widgets.VBox([self.model_upload_button, self.model_remove_button,self.spc1]), self.spc1,widgets.VBox([self.model_multi_select_upload_output,self.spc2, 
                                                        widgets.HBox([self.spc1,self.spc1,self.spc2,self.model_name_label,self.model_rename_button])])])

        
        # Screen to choose the lat_lon type data to be linked to the models.
        self.model_data_select_multiple = widgets.SelectMultiple(options=[self.model_data_multi_select_dict_show[mdl] for mdl in self.data_list],description='Data:',layout=widgets.Layout(width="380px",height="180px"))
        self.model_data_select_multiple.observe(self.on_change_model_data_ml, names = 'value')
        self.model_data_select_multiple_output = out(self.model_data_select_multiple)

        # Screen to choose the models to be linked to the models.
        self.model_select = widgets.Select(options=self.model_list + [''],description='Models:',value='',layout=widgets.Layout(width="380px",height="180px"))
        self.model_select.observe(self.on_change_model_ml, names = 'value')
        self.model_select_output = out(self.model_select)
        
        # Model link button
        self.model_link_button = widgets.Button(description="Link", layout=widgets.Layout(width="80px"))
        self.model_link_button.style.button_color = "lightgray"
        self.model_link_button.on_click(self.link_model)
        
        # Model unlink button
        self.model_unlink_button = widgets.Button(description="Unlink", layout=widgets.Layout(width="80px"))
        self.model_unlink_button.style.button_color = "lightgray"
        self.model_unlink_button.on_click(self.unlink_model)
        
        self.model_link_box = widgets.VBox([widgets.HBox([self.model_link_button,widgets.Label("",layout=widgets.Layout(width="1%")),self.model_unlink_button]), self.spc2,
                                                widgets.HBox([self.model_select_output,self.spc2,self.model_data_select_multiple_output])])

        
        # If embedding_list is empty value element is empty
        if(len(self.embedding_list)>0):
            self.embedding_multi_select_upload = widgets.SelectMultiple(options=self.embedding_list,description='Emb:', value=[self.embedding_list[0]],layout=widgets.Layout(width="380px",height="180px"))

        else:       
            self.embedding_multi_select_upload = widgets.SelectMultiple(options=self.embedding_list,description='Emb:', value=[],layout=widgets.Layout(width="380px",height="180px"))
        
        # the screen that shows the embedding that are uploaded to the tool
        self.embedding_multi_select_upload_output = out(self.embedding_multi_select_upload)

        # Embedding upload button
        self.embedding_upload_button = widgets.FileUpload(acept=".csv",multiple=True,layout=widgets.Layout(width="150px"))
        self.embedding_upload_button.observe(self.upload_embedding, names="value")
        
        # Embedding remove button
        self.remove_embedding_button = widgets.Button(description="Remove", layout=widgets.Layout(width="150px"))
        self.remove_embedding_button.on_click(self.remove_embedding)
        
        # Embedding rename button
        self.rename_embedding_button = widgets.Button(description="Rename", layout=widgets.Layout(width="90px"))
        self.rename_embedding_button.style.button_color = "lightgray"
        self.rename_embedding_button.on_click(self.rename_embedding)
        
        # The space to put the name of embedding to be renamed.
        self.embedding_name_label = widgets.Text(description='',layout=widgets.widgets.Layout(width="250px"))
        self.embedding_name_label.observe(self.on_change_emb_name, names = 'value')
        
        self.embedding_upload_box = widgets.HBox(children=[widgets.VBox([self.embedding_upload_button,self.remove_embedding_button,self.spc1]),self.spc1,
                                widgets.VBox([self.embedding_multi_select_upload_output,self.spc2, widgets.HBox([self.spc1,self.spc1,self.spc2,self.embedding_name_label,self.rename_embedding_button])])])
        
        # Screen to choose the lat_lon type data to be linked to the embedding.
        self.embedding_data_select_multiple = widgets.SelectMultiple(options=[self.data_multi_select_dict_show[dl] for dl in self.data_list],description='Data:', layout=widgets.Layout(width="380px",height="180px"))
        self.embedding_data_select_multiple.observe(self.on_change_emb_data_ml, names = 'value')
        self.embedding_data_select_multiple_output = out(self.embedding_data_select_multiple)
        
        # Screen to choose the embeddings to be linked to the embeddings.
        self.embedding_select = widgets.Select(options=self.embedding_list + [''],description='Emb:',value='',layout=widgets.Layout(width="380px",height="180px"))
        self.embedding_select.observe(self.on_change_emb_ml, names = 'value')
        self.embedding_select_output = out(self.embedding_select)
        
        
        # Embedding link button
        self.embedding_link_button = widgets.Button(description="Link", layout=widgets.Layout(width="80px"))
        self.embedding_link_button.style.button_color = "lightgray"
        self.embedding_link_button.on_click(self.link_embedding)

        # Embedding unlink button
        self.embedding_unlink_button = widgets.Button(description="Unlink", layout=widgets.Layout(width="80px"))
        self.embedding_unlink_button.style.button_color = "lightgray"
        self.embedding_unlink_button.on_click(self.unlink_embedding)
        
        
        
        self.embedding_link_box = widgets.VBox([widgets.HBox([self.embedding_link_button,widgets.Label("",layout=widgets.Layout(width="1%")),self.embedding_unlink_button]), 
                        self.spc2,widgets.HBox([self.embedding_select_output, self.spc2, self.embedding_data_select_multiple_output])])


        self.tab_model = widgets.Tab()
        self.tab_model.children = [self.model_upload_box,self.model_link_box]
        self.tab_model.set_title(0,'Upload')
        self.tab_model.set_title(1,'Data Linkage')

        self.tab_embedding = widgets.Tab()
        self.tab_embedding.children = [self.embedding_upload_box ,self.embedding_link_box]
        self.tab_embedding.set_title(0,'Upload')
        self.tab_embedding.set_title(1,'Data Linkage')

        self.upload_accordion = widgets.Accordion(children=[self.data_upload_box,self.tab_embedding,self.tab_model],selected_index=None)
        self.upload_accordion.set_title(0, 'Datasets Import')
        self.upload_accordion.set_title(1, 'Embeddings Import')
        self.upload_accordion.set_title(2, 'Models Import')
          
       

    def on_change_data_name (self, change):

        self.data_name = change.new

    def on_change_model_name (self, change):

        self.model_name = change.new

    def on_change_model_data_ml (self, change):
    
        self.model_data_multi_select = change.new

    def on_change_model_ml(self, change):
        
        self.model_multi_select = change.new
        
        model_data = pd.read_csv("Model#Data.csv")
           
        if(self.model_multi_select in list(model_data["Model"])):
            
            data_link_string = model_data[model_data["Model"] == self.model_multi_select]["Data"].values[0].split("%")
            aux_folder = listdir("data/")
            if('.csv' in aux_folder):
                aux_folder.remove('.csv')
            data_folder = [data.split(".")[0] for data in aux_folder]

            with self.model_data_select_multiple_output:

                self.model_data_select_multiple_output.clear_output()
                
                for data in data_folder:
                    
                    if(data in data_link_string):
                    
                        self.model_data_multi_select_dict_show[data] = data + " " +"\u2714" 
                    
                    else:
                        
                        self.model_data_multi_select_dict_show[data] = data
                
                self.model_data_select_multiple.options = [self.model_data_multi_select_dict_show[mdl] for mdl in self.data_list]
                display(self.model_data_select_multiple)
            
        else:

            with self.model_data_select_multiple_output:

                self.model_data_select_multiple_output.clear_output()
                self.model_data_select_multiple.options = [mdl for mdl in self.data_list]
                
                for mdl in self.data_list:
                    self.model_data_multi_select_dict_show[mdl] = mdl
                
                display(self.model_data_select_multiple)

    def on_change_emb_data_ml (self, change):

        self.embedding_data_multi_select = change.new

    def on_change_emb_name (self, change):

        self.embedding_name = change.new

    def on_change_emb_ml (self, change): 
    
        self.embedding_multi_select = change.new
        
        emb_data = pd.read_csv("Emb#Data.csv")

        # Checks if the selected embedding is in the links table, if it is, it makes a change to the screen that shows dataset of type lat_lon, showing the linked datasets with the check symbol.
        
        if(self.embedding_multi_select in list(emb_data["Emb"])):
            
            data_link_string = emb_data[emb_data["Emb"] == self.embedding_multi_select]["Data"].values[0].split("%")
            aux_folder = listdir("data/")
            if('.csv' in aux_folder):
                aux_folder.remove('.csv')
            data_folder = [data.split(".")[0] for data in aux_folder]

            with self.embedding_data_select_multiple_output:

                self.embedding_data_select_multiple_output.clear_output()
                
                for data in data_folder:
                    
                    if(data in data_link_string):
                    
                        self.data_multi_select_dict_show[data] = data + " " +"\u2714" 
                    
                    else:
                        
                        self.data_multi_select_dict_show[data] = data
                
                self.embedding_data_select_multiple.options = [self.data_multi_select_dict_show[dl] for dl in self.data_list]
                display(self.embedding_data_select_multiple)
            
        else:

            with self.embedding_data_select_multiple_output:

                self.embedding_data_select_multiple_output.clear_output()
                self.embedding_data_select_multiple.options = [edl for edl in self.data_list]
                display(self.embedding_data_select_multiple)


    def upload_data(self, data_upload_button):

        
        with self.data_upload_output:
                
            self.data_upload_output.clear_output()
            
            display(widgets.Label("Processing..."))
            
            new_values = data_upload_button.new
            keys = list(new_values.keys())

            for k in keys:
                data = StringIO(str(new_values[k]["content"],'utf-8'))
                df=pd.read_csv(data,index_col=0)
                df.to_csv("data/"+k.split(".")[0]+".csv")
                
            self.data_list = [data.split(".")[0] for data in listdir("data/")]
            
            if("" in self.data_list):
                self.data_list.remove("")
            
            self.data_upload.options= self.data_list 
            self.data_upload.value= [self.data_list[0]]
            
            self.data_upload_output.clear_output()
            
            display(self.data_upload)

        # When uploading new data sets of type lat_lon, the screens to be linked to embeddings and models are modified as well as the Dropdown of the trajectory view tab.

        with self.embedding_data_select_multiple_output :

            self.embedding_data_select_multiple_output.clear_output()
        
            keys = list(new_values.keys())
            
            for k in keys:
                self.data_multi_select_dict_show[k.split(".")[0]] = k.split(".")[0]

            
            self.embedding_data_select_multiple.options = [self.data_multi_select_dict_show[dl] for dl in self.data_list]
            display(self.embedding_data_select_multiple)

        with self.model_data_select_multiple_output:

            self.model_data_select_multiple_output.clear_output()
        
            keys = list(new_values.keys())
            
            for k in keys:
                self.model_data_multi_select_dict_show[k.split(".")[0]] = k.split(".")[0]

            
            self.model_data_select_multiple.options = [self.model_data_multi_select_dict_show[mdl] for mdl in self.data_list]
            display(self.model_data_select_multiple)

        with self.VTDO.select_data_dropdown_output:

            self.VTDO.select_data_dropdown_output.clear_output()
            self.VTDO.select_data_dropdown = widgets.Dropdown(description='',layout=widgets.Layout(width="80px"),options=self.data_list)
            self.VTDO.select_data_dropdown.observe(self.VTDO.on_change_dataset, names='value')
            display(self.VTDO.select_data_dropdown)

    def remove_data(self, data_remove_button):

        emb_data = pd.read_csv("Emb#Data.csv")
        model_data = pd.read_csv("Model#Data.csv")
        
        dup_val = self.data_upload.value
        aux = []
        
        for d in self.data_list:
            if(not(d in dup_val)):
                aux.append(d)
            
        self.data_list = aux
        
        with self.data_upload_output:
            
            self.data_upload_output.clear_output()
            
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

                for ID in model_data.index:
                        
                    L_data = model_data.loc[ID].Data.split("%")
                    
                    if(d in L_data):
                        
                        L_data.remove(d)
                        
                        if(len(L_data)==0):
                            
                            model_data = model_data.drop(model_data[model_data.Model == model_data.loc[ID].Model].index)
                            
                        else:
                            
                            model_data.loc[ID,"Data"] = "%".join(L_data)
                
            emb_data.to_csv("Emb#Data.csv",index=False)
            model_data.to_csv("Model#Data.csv",index=False)
                        
            
            if(len(self.data_list)>0):

                self.data_upload = widgets.SelectMultiple(options=self.data_list,description='Data:', value=[self.data_list[0]],layout=widgets.Layout(width="380px",height="180px"))
            else:
        
                self.data_upload = widgets.SelectMultiple(options=self.data_list,description='Data:', value=[],layout=widgets.Layout(width="380px",height="180px"))
            
            display(self.data_upload)

        # When a data_set of type lat_lon is removed, the link between it and the embedding is lost, so, in addition to removing this data from the screen of 
        # links and models and embeddings, this data disappears from the Dropdown of the view trajectory tab

        with self.embedding_data_select_multiple_output:

            self.embedding_data_select_multiple_output.clear_output()
            
            for ex in dup_val:
                self.data_multi_select_dict_show.pop(ex)
            
            aux_list_2 = [self.data_multi_select_dict_show[dl] for dl in self.data_list]
            self.embedding_data_select_multiple = widgets.SelectMultiple(options=aux_list_2,description='Data:', layout=widgets.Layout(width="380px",height="180px"))
            self.embedding_data_select_multiple.observe(self.on_change_emb_data_ml, names = 'value')
            #self.embedding_data_select_multiple.value = [aux_list_2]
            display(self.embedding_data_select_multiple)
            
        with self.model_data_select_multiple_output:

            self.model_data_select_multiple_output.clear_output()
            
            for ex in dup_val:
                self.model_data_multi_select_dict_show.pop(ex)
            
            aux_list_1 = [self.model_data_multi_select_dict_show[mdl] for mdl in self.data_list]
            self.model_data_select_multiple = widgets.SelectMultiple(options=aux_list_1,description='Data:', layout=widgets.Layout(width="380px",height="180px"))
            self.model_data_select_multiple.observe(self.on_change_model_data_ml, names = 'value')
            #self.model_data_select_multiple.value = [aux_list_1[0]]
            display(self.model_data_select_multiple)

        with self.EEO.select_mutiples_models_output:

            self.EEO.select_mutiples_models_output.clear_output()

            self.Model_df  = pd.read_csv("Model#Data.csv")
            self.Model_dict = {md:dt.split("%")[0] for md,dt in zip(self.Model_df.Model,self.Model_df.Data)}
            self.models_names = list(self.Model_df.Model)

            self.EEO.select_mutiples_models = widgets.SelectMultiple( options=self.models_names + [''], value=[''] , description='Models:', layout=widgets.Layout(width="420px",height="140px"), disabled=False) 

            display(self.EEO.select_mutiples_models)
            
            
        with self.VTDO.select_data_dropdown_output:

            self.VTDO.select_data_dropdown_output.clear_output()
            self.VTDO.select_data_dropdown = widgets.Dropdown(description='',layout=widgets.Layout(width="80px"),options=self.data_list)
            self.VTDO.select_data_dropdown.observe(self.VTDO.on_change_dataset, names='value')
            display(self.VTDO.select_data_dropdown)

        with self.IEO.embedding_choice_out:
        
            self.IEO.Emb_list = list(pd.read_csv("Emb.csv").Emb)
            self.IEO.Emb_df = pd.read_csv("Emb#Data.csv") 
            self.IEO.Emb_dict = { emb: self.IEO.Emb_df[self.IEO.Emb_df["Emb"]==emb].Data.values[0].split("%") for emb in list(self.IEO.Emb_df.Emb)}
            self.IEO.embedding_choice.options = self.IEO.Emb_list 
            self.IEO.embedding_choice.value = self.IEO.Emb_list[0]
            self.IEO.embedding_choice_out.clear_output()
            display(self.IEO.embedding_choice)
            
        with self.IEO.text_dataset_out:
                  
            self.IEO.text_dataset.options =  [''] if(not(self.IEO.Emb_list[0] in self.IEO.Emb_dict.keys())) else self.IEO.Emb_dict[self.IEO.Emb_list[0]] + ['']
            self.IEO.text_dataset.value = ''
            self.IEO.text_dataset_out.clear_output()  
            display(self.IEO.text_dataset)
            
    def rename_data(self, data_rename_button):


        emb_data = pd.read_csv("Emb#Data.csv")
        model_data = pd.read_csv("Model#Data.csv")
        
        dup_val = self.data_upload.value
        aux = []
        
        for d in self.data_list:
            if(d in dup_val):
                aux.append(self.data_name)
            else:
                aux.append(d)
                
        self.data_list = aux
        
        with self.data_upload_output:
            
            self.data_upload_output.clear_output()
            
            for d in dup_val:
                
                os.rename("data/"+d+".csv","data/"+self.data_name+".csv")
            
                for ID in emb_data.index:

                    L_data = emb_data.loc[ID].Data.split("%")

                    aux_ld =[]

                    for ld in L_data:

                        if(ld==d):
                            aux_ld.append(self.data_name)
                        else:
                            aux_ld.append(ld)

                    L_data = aux_ld                   

                    emb_data.loc[ID,"Data"] = "%".join(L_data)

                for ID in model_data.index:

                    L_data = model_data.loc[ID].Data.split("%")

                    aux_ld =[]

                    for ld in L_data:

                        if(ld==d):
                            aux_ld.append(self.data_name)
                        else:
                            aux_ld.append(ld)

                    L_data = aux_ld                   

                    model_data.loc[ID,"Data"] = "%".join(L_data)


            emb_data.to_csv("Emb#Data.csv",index=False)
            model_data.to_csv("Model#Data.csv",index=False)
                
            
            if(len(self.data_list)>0):

                self.data_upload = widgets.SelectMultiple(options=self.data_list,description='Data:', value=[self.data_list[0]],layout=widgets.Layout(width="380px",height="180px"))
            else:
        
                self.data_upload = widgets.SelectMultiple(options=self.data_list,description='Data:', value=[],layout=widgets.Layout(width="380px",height="180px"))
            
            display(self.data_upload)

        with  self.embedding_data_select_multiple_output:

            self.embedding_data_select_multiple_output.clear_output()
        
            for ex in dup_val:
                
                if("\u2714" in self.data_multi_select_dict_show[ex]):
                    
                    self.data_multi_select_dict_show[self.data_name] = self.data_name  + " " +"\u2714" 
                    
                else:
                    
                    self.data_multi_select_dict_show[self.data_name] = self.data_name
                    
                self.data_multi_select_dict_show.pop(ex)

            self.embedding_data_select_multiple = widgets.SelectMultiple(options=[self.data_multi_select_dict_show[edl] for edl in self.data_list],description='Data:', layout=widgets.Layout(width="380px",height="180px"))
            self.embedding_data_select_multiple.observe(self.on_change_emb_data_ml, names = 'value')
            display(self.embedding_data_select_multiple) 
            
        with self.model_data_select_multiple_output:

            self.model_data_select_multiple_output.clear_output()
            
            for ex in dup_val:
                
                if("\u2714" in self.model_data_multi_select_dict_show[ex]):
                    
                    self.model_data_multi_select_dict_show[self.data_name] = self.data_name + " " +"\u2714" 
                    
                else:
                    
                    self.model_data_multi_select_dict_show[self.data_name] = self.data_name
                    
                self.model_data_multi_select_dict_show.pop(ex)

            model_data_ml = widgets.SelectMultiple(options=[self.model_data_multi_select_dict_show[mdl] for mdl in self.data_list],description='Data:', layout=widgets.Layout(width="380px",height="180px"))
            model_data_ml.observe(self.on_change_model_data_ml, names = 'value')
            display(model_data_ml)

        with self.EEO.select_mutiples_models_output:

            self.EEO.select_mutiples_models_output.clear_output()

            self.Model_df  = pd.read_csv("Model#Data.csv")
            self.Model_dict = {md:dt.split("%")[0] for md,dt in zip(self.Model_df.Model,self.Model_df.Data)}
            self.models_names = list(self.Model_df.Model)

            self.EEO.select_mutiples_models = widgets.SelectMultiple( options=self.models_names + [''], value=[''] , description='Models:', layout=widgets.Layout(width="420px",height="140px"), disabled=False) 

            display(self.EEO.select_mutiples_models)
    
         
        with self.VTDO.select_data_dropdown_output:

            self.VTDO.select_data_dropdown_output.clear_output()
            self.VTDO.select_data_dropdown = widgets.Dropdown(description='',layout=widgets.Layout(width="80px"),options=self.data_list)
            self.VTDO.select_data_dropdown.observe(self.VTDO.on_change_dataset, names='value')
            display(self.VTDO.select_data_dropdown)

        with self.IEO.embedding_choice_out:
            
            self.IEO.Emb_list = list(pd.read_csv("Emb.csv").Emb)
            self.IEO.Emb_df = pd.read_csv("Emb#Data.csv") 
            self.IEO.Emb_dict = { emb: self.IEO.Emb_df[self.IEO.Emb_df["Emb"]==emb].Data.values[0].split("%") for emb in list(self.IEO.Emb_df.Emb)}
            self.IEO.embedding_choice.options = self.IEO.Emb_list 
            self.IEO.embedding_choice.value = self.IEO.Emb_list[0]
            self.IEO.embedding_choice_out.clear_output()
            display(self.IEO.embedding_choice)
            


    def upload_model(self, model_upload_button):
            
        with self.model_multi_select_upload_output:
                
            
            self.model_multi_select_upload_output.clear_output()
            
            display(widgets.Label("Processing..."))
            
            new_values = model_upload_button.new
            keys = list(new_values.keys())
            
            Emb_list = list(pd.read_csv("Emb.csv")["Emb"])
            Model_Emb_df = pd.read_csv("Model#Emb.csv")

            for k in keys:
                data= io.BytesIO(new_values[k]["content"])
                model= h5py.File(data,'r')
                model_keras= load_model(model)
                model_keras.save("Models/"+k.split(".")[0]+".h5")
                
                for emb in Emb_list: 
                
                    if(not(k.split(".")[0] in list(Model_Emb_df.Model) )):

                        if(k.split(".")[0] == emb.split("_")[0]):

                            Model_Emb_df = pd.DataFrame({"Model":list(Model_Emb_df["Model"]) + [k.split(".")[0]] , 
                                        "Emb":list(Model_Emb_df["Emb"]) + [emb]})
            
            
            Model_Emb_df.to_csv("Model#Emb.csv",index=False)
                        
                
            self.model_list = [model.split(".")[0] for model in listdir("models/")]
            
            if("" in self.model_list):
                self.model_list.remove("")
            
            pd.DataFrame({"Model":self.model_list}).to_csv("Model.csv",index=False)
            
            self.model_multi_select_upload.options= self.model_list 
            self.model_multi_select_upload.value = [self.model_list[0]]
            
            self.model_multi_select_upload_output.clear_output()
            
            display(self.model_multi_select_upload)
            
        with self.model_select_output:

            self.model_select_output.clear_output()
            self.model_select = widgets.Select(options=self.model_list + [''],description='Models:',value='',layout=widgets.Layout(width="380px",height="180px"))
            self.model_select.observe(self.on_change_model_ml, names = 'value') 
            display(self.model_select)
        
    def remove_model(self, model_remove_button):    
    
        model_data = pd.read_csv("Model#Data.csv")
        model_emb_df = pd.read_csv("Model#Emb.csv")
        models_df = pd.read_csv("Model.csv")
        
        mmlup_val = self.model_multi_select_upload.value
        aux = []
        
        for ml in self.model_list:
            if(not(ml in mmlup_val)):
                aux.append(ml)
                
        self.model_list = aux
        
        with self.model_multi_select_upload_output:
            
            self.model_multi_select_upload_output.clear_output()
            
            for mv in mmlup_val: 
                
                os.remove("models/"+mv+".h5")
                models_df = models_df.drop(models_df[models_df.Model == mv].index)
                model_data = model_data.drop(model_data[model_data.Model == mv].index)
                model_emb_df = model_emb_df.drop(model_emb_df[model_emb_df.Model == mv].index)
                
            
            models_df.to_csv("Model.csv",index=False)
            model_data.to_csv("Model#Data.csv",index=False)
            model_emb_df.to_csv("Model#Emb.csv",index=False)
            
            if(len(self.model_list)>0):
                    
                self.model_multi_select_upload = widgets.SelectMultiple(options= self.model_list, description='Models:', value=[self.model_list[0]], layout=widgets.Layout(width="380px",height="180px") )
                
            else:
                    
                self.model_multi_select_upload = widgets.SelectMultiple(options= self.model_list,description='Models:', value=[], layout=widgets.Layout(width="380px",height="180px"))
            
            display(self.model_multi_select_upload)
            
        with self.model_select_output:

            self.model_select_output.clear_output()
            self.model_select.options = self.model_list + ['']
            self.model_select.value = ''
            
            display(self.model_select)

        with self.EEO.select_mutiples_models_output:

            self.EEO.select_mutiples_models_output.clear_output()

            self.Model_df  = pd.read_csv("Model#Data.csv")
            self.Model_dict = {md:dt.split("%")[0] for md,dt in zip(self.Model_df.Model,self.Model_df.Data)}
            self.models_names = list(self.Model_df.Model)

            self.EEO.select_mutiples_models = widgets.SelectMultiple( options=self.models_names + [''], value=[''] , description='Models:', layout=widgets.Layout(width="420px",height="140px"), disabled=False) 

            display(self.EEO.select_mutiples_models)
        
    def rename_model(self, model_rename_button):
        
        model_data = pd.read_csv("Model#Data.csv")
        models_df = pd.read_csv("Model.csv")
        model_emb_df = pd.read_csv("Model#Emb.csv")
        
        mmlup_val = self.model_multi_select_upload.value
        aux = []
        
        
        for ml in self.model_list:
            if(ml in mmlup_val):
                aux.append(self.model_name)
            else:
                aux.append(ml)
                
        self.model_list = aux
        
        with self.model_multi_select_upload_output:
            
            self.model_multi_select_upload_output.clear_output()
            
            for mv in mmlup_val: 
                
                os.rename("models/"+mv+".h5", "models/"+self.model_name+".h5")
                models_df.replace(mv,self.model_name,inplace=True)
                model_data.replace(mv,self.model_name,inplace=True)
                model_emb_df.replace(mv,self.model_name,inplace=True)
                
            
            models_df.to_csv("Model.csv",index=False)
            model_data.to_csv("Model#Data.csv",index=False)
            model_emb_df.to_csv("Model#Emb.csv",index=False)
                    

            if(len(self.model_list)>0):

                self.model_multi_select_upload = widgets.SelectMultiple(options=self.model_list,description='Models:', value=[self.model_list[0]],layout=widgets.Layout(width="380px",height="180px"))
            else:
        
                self.model_multi_select_upload = widgets.SelectMultiple(options=self.model_list,description='Models:', value=[],layout=widgets.Layout(width="380px",height="180px"))
            
            display(self.model_multi_select_upload)
            
        with self.model_select_output:

            self.model_select_output.clear_output()
            self.model_select = widgets.Select(options=self.model_list + [''],description='Models:',value='',layout=widgets.Layout(width="380px",height="180px"))
            self.model_select.observe(self.on_change_model_ml, names = 'value')
            self.model_select.value = self.model_list[0]
            display(self.model_select)

        with self.EEO.select_mutiples_models_output:

            self.EEO.select_mutiples_models_output.clear_output()

            self.Model_df  = pd.read_csv("Model#Data.csv")
            self.Model_dict = {md:dt.split("%")[0] for md,dt in zip(self.Model_df.Model,self.Model_df.Data)}
            self.models_names = list(self.Model_df.Model)

            self.EEO.select_mutiples_models = widgets.SelectMultiple( options=self.models_names + [''], value=[''] , description='Models:', layout=widgets.Layout(width="420px",height="140px"), disabled=False) 

            display(self.EEO.select_mutiples_models)
            

    def link_model(self, model_link_button):

 
        model = self.model_select.value

        data = [ mml.split(" ")[0] if("\u2714" in mml) else mml for mml in self.model_data_select_multiple.value]
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

            
            with self.model_data_select_multiple_output:
                
                self.model_data_select_multiple_output.clear_output()
                
                for d in data_folder:
                    
                    if(d in data):
                        self.model_data_multi_select_dict_show[d] = d + " " +"\u2714" 
                        
                    else:
                        self.model_data_multi_select_dict_show[d] = d 
                
                self.model_data_select_multiple.options = [self.model_data_multi_select_dict_show[mdl] for mdl in self.data_list]
                display(self.model_data_select_multiple)
            
        else:
            
        
            MD_df = pd.DataFrame({"Model":list(model_data["Model"]) + [model] , "Data":list(model_data["Data"]) + ["%".join(data)]})
            
            MD_df.to_csv("Model#Data.csv",index=False)
            
            with self.model_data_select_multiple_output:
                
                self.model_data_select_multiple_output.clear_output()
                
                for d in data_folder:
                    
                    if(d in data):
                        self.model_data_multi_select_dict_show[d] = d + " " +"\u2714" 
                        
                    else:
                        self.model_data_multi_select_dict_show[d] = d  
                
                self.model_data_select_multiple.options = [self.model_data_multi_select_dict_show[mdl] for mdl in self.data_list]
                display(self.model_data_select_multiple)

        with self.EEO.select_mutiples_models_output:

            self.EEO.select_mutiples_models_output.clear_output()

            self.Model_df  = pd.read_csv("Model#Data.csv")
            self.Model_dict = {md:dt.split("%")[0] for md,dt in zip(self.Model_df.Model,self.Model_df.Data)}
            self.models_names = list(self.Model_df.Model)

            self.EEO.select_mutiples_models = widgets.SelectMultiple( options=self.models_names + [''], value=[''] , description='Models:', layout=widgets.Layout(width="420px",height="140px"), disabled=False) 

            display(self.EEO.select_mutiples_models)
                 
    def unlink_model(self, model_unlink_button):
    
        model = self.model_select.value
        data = [ mml.split(" ")[0] if("\u2714" in mml) else mml for mml in self.model_data_select_multiple.value]
        
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

            
        with self.model_data_select_multiple_output:

            self.model_data_select_multiple_output.clear_output()

            for d in data:
                self.model_data_multi_select_dict_show[d] = d

            self.model_data_select_multiple.options = [self.model_data_multi_select_dict_show[mdl] for mdl in self.data_list]
            display(self.model_data_select_multiple)

        with self.EEO.select_mutiples_models_output:

            self.EEO.select_mutiples_models_output.clear_output()

            self.Model_df  = pd.read_csv("Model#Data.csv")
            self.Model_dict = {md:dt.split("%")[0] for md,dt in zip(self.Model_df.Model,self.Model_df.Data)}
            self.models_names = list(self.Model_df.Model)

            self.EEO.select_mutiples_models = widgets.SelectMultiple( options=self.models_names + [''], value=[''] , description='Models:', layout=widgets.Layout(width="420px",height="140px"), disabled=False) 

            display(self.EEO.select_mutiples_models)
       
        
    

    def upload_embedding(self, embedding_upload_button):
    
        with self.embedding_multi_select_upload_output:
                
            self.embedding_multi_select_upload_output.clear_output()
            
            display(widgets.Label("Processing..."))
            
            new_values = embedding_upload_button.new
            keys = list(new_values.keys())

            for k in keys:
                
                data = StringIO(str(new_values[k]["content"],'utf-8'))
                df=pd.read_csv(data,index_col=0)
                df.to_csv("embeddings/"+k.split(".")[0]+".csv")
                
            self.embedding_list = [emb.split(".")[0] for emb in listdir("embeddings/")]
            
            if("" in self.embedding_list):
                self.embedding_list.remove("")
            
            pd.DataFrame({"Emb":self.embedding_list}).to_csv("Emb.csv",index=False)
            
            self.embedding_multi_select_upload.options = self.embedding_list
            self.embedding_multi_select_upload .value = [self.embedding_list[0]]
            
            self.embedding_multi_select_upload_output.clear_output()
            
            display(self.embedding_multi_select_upload)
            
        with self.embedding_select_output:

            self.embedding_select_output.clear_output()
            self.embedding_select = widgets.Select(options=self.embedding_list + [''],description='Embeddings:',value='',layout=widgets.Layout(width="380px",height="180px"))
            self.embedding_select.observe(self.on_change_emb_ml, names = 'value')    
            display(self.embedding_select)   

        with self.IEO.embedding_choice_out:
            
            self.IEO.Emb_list = list(pd.read_csv("Emb.csv").Emb)
            self.IEO.Emb_df = pd.read_csv("Emb#Data.csv") 
            self.IEO.Emb_dict = { emb: self.IEO.Emb_df[self.IEO.Emb_df["Emb"]==emb].Data.values[0].split("%") for emb in list(self.IEO.Emb_df.Emb)}
            self.IEO.embedding_choice.options = self.IEO.Emb_list 
            self.IEO.embedding_choice.value = self.IEO.Emb_list[0]
            self.IEO.embedding_choice_out.clear_output()
            display(self.IEO.embedding_choice)
        
    def remove_embedding(self, remove_embedding_button):
    
        emb_data = pd.read_csv("Emb#Data.csv")
        emb_df = pd.read_csv("Emb.csv")
        
        emlup_val = self.embedding_multi_select_upload.value
        aux = []

        for emb in self.embedding_list:
            if(not(emb in emlup_val)):
                aux.append(emb)
                
        
        self.embedding_list = aux
        
        with self.embedding_multi_select_upload_output:
            self.embedding_multi_select_upload_output.clear_output()
            
            for emb in emlup_val: 
                
                os.remove("embeddings/"+emb+".csv")
                emb_df = emb_df.drop(emb_df[emb_df.Emb == emb].index)
                emb_data = emb_data.drop(emb_data[emb_data.Emb == emb].index)
                
            
            emb_df.to_csv("Emb.csv",index=False)
            emb_data.to_csv("Emb#Data.csv",index=False)
            
            
            
            if(len(self.embedding_list)>0):
                self.embedding_multi_select_upload = widgets.SelectMultiple(options= self.embedding_list,description='Emb:', value=[self.embedding_list[0]],layout=widgets.Layout(width="380px",height="180px"))
                
            else:
                self.embedding_multi_select_upload = widgets.SelectMultiple(options= self.embedding_list,description='Emb:', value=[],layout=widgets.Layout(width="380px",height="180px"))
            
            display(self.embedding_multi_select_upload)
            
        with self.embedding_select_output:

            self.embedding_select_output.clear_output()
            self.embedding_select.options = self.embedding_list + ['']
            self.embedding_select.value = ''
            display(self.embedding_select)

        with self.IEO.embedding_choice_out:
            
            self.IEO.Emb_list = list(pd.read_csv("Emb.csv").Emb)
            self.IEO.Emb_df = pd.read_csv("Emb#Data.csv") 
            self.IEO.Emb_dict = { emb: self.IEO.Emb_df[self.IEO.Emb_df["Emb"]==emb].Data.values[0].split("%") for emb in list(self.IEO.Emb_df.Emb)}
            self.IEO.embedding_choice.options = self.IEO.Emb_list 
            self.IEO.embedding_choice.value = self.IEO.Emb_list[0]
            self.IEO.embedding_choice_out.clear_output()
            display(self.IEO.embedding_choice)

    def rename_embedding(self, rename_embedding_button):    
                    
        
        emb_data = pd.read_csv("Emb#Data.csv")
        emb_df = pd.read_csv("Emb.csv")
        
        emlup_val = self.embedding_multi_select_upload.value
        aux = []
        
        
        for emb in self.embedding_list:

            if(emb in emlup_val):
                aux.append(self.embedding_name)
            else:
                aux.append(emb)
                
        self.embedding_list = aux
        
        with self.embedding_multi_select_upload_output:
            
            self.embedding_multi_select_upload_output.clear_output()
            
            for emb in emlup_val: 
                
                os.rename("embeddings/"+emb+".csv", "embeddings/"+self.embedding_name+".csv")
                emb_df.replace(emb,self.embedding_name,inplace=True)
                emb_data.replace(emb,self.embedding_name,inplace=True)
                
            
            emb_df.to_csv("Emb.csv",index=False)
            emb_data.to_csv("Emb#Data.csv",index=False)
            
            if(len(self.embedding_list)>0):

                self.embedding_multi_select_upload = widgets.SelectMultiple(options= self.embedding_list, description='Emb:', value=[self.embedding_list[0]],layout=widgets.Layout(width="380px",height="180px"))
            else:
        
                self.embedding_multi_select_upload = widgets.SelectMultiple(options= self.embedding_list, description='Emb:', value=[],layout=widgets.Layout(width="380px",height="180px"))
                    
            display(self.embedding_multi_select_upload)
            
        with self.embedding_select_output:

            self.embedding_select_output.clear_output()
            self.embedding_select = widgets.Select(options= self.embedding_list + [''],description='Embeddings:',value='',layout=widgets.Layout(width="380px",height="180px"))
            self.embedding_select.observe(self.on_change_emb_ml, names = 'value')
            #self.embedding_select.value = self.embedding_list[0]
            display(self.embedding_select)

        with self.IEO.embedding_choice_out:
            
            self.IEO.Emb_list = list(pd.read_csv("Emb.csv").Emb)
            self.IEO.Emb_df = pd.read_csv("Emb#Data.csv") 
            self.IEO.Emb_dict = { emb: self.IEO.Emb_df[self.IEO.Emb_df["Emb"]==emb].Data.values[0].split("%") for emb in list(self.IEO.Emb_df.Emb)}
            self.IEO.embedding_choice.options = self.IEO.Emb_list 
            self.IEO.embedding_choice.value = self.IEO.Emb_list[0]
            self.IEO.embedding_choice_out.clear_output()
            display(self.IEO.embedding_choice)


                         
    def link_embedding(self, embedding_link_button):
        
        emb = self.embedding_select.value
        data = [ eml.split(" ")[0] if("\u2714" in eml) else eml for eml in self.embedding_data_select_multiple.value]
        
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
            
            with self.embedding_data_select_multiple_output:
                
                self.embedding_data_select_multiple_output.clear_output()
                
                for d in data_folder:
                    
                    if(d in data):
                        self.data_multi_select_dict_show[d] = d + " " +"\u2714" 
                        
                    else:
                        self.data_multi_select_dict_show[d] = d 
                    
                self.embedding_data_select_multiple.options = [self.data_multi_select_dict_show[dl] for dl in self.data_list]
                display(self.embedding_data_select_multiple)
            
        else:
            
        
            EB_df = pd.DataFrame({"Emb":list(emb_data["Emb"]) + [emb] , "Data":list(emb_data["Data"]) + ["%".join(data)]})
            
            EB_df.to_csv("Emb#Data.csv",index=False)
            
            with self.embedding_data_select_multiple_output:
                
                self.embedding_data_select_multiple_output.clear_output()
                
                for d in data_folder:
                    
                    if(d in data):
                        self.data_multi_select_dict_show[d] = d + " " +"\u2714" 
                        
                    else:
                        self.data_multi_select_dict_show[d] = d  
                
                self.embedding_data_select_multiple.options = [self.data_multi_select_dict_show[dl] for dl in self.data_list]
                display(self.embedding_data_select_multiple)



        with self.IEO.text_dataset_out:
            
            self.IEO.text_dataset_out.clear_output()
            self.IEO.text_dataset.options =  [''] if(not(self.IEO.Emb_list[0] in self.IEO.Emb_dict.keys())) else self.IEO.Emb_dict[self.IEO.Emb_list[0]] + ['']
            self.IEO.text_dataset.value = ''
            display(self.IEO.text_dataset)

        with self.IEO.embedding_choice_out:
            
            self.IEO.Emb_list = list(pd.read_csv("Emb.csv").Emb)
            self.IEO.Emb_df = pd.read_csv("Emb#Data.csv") 
            self.IEO.Emb_dict = { emb: self.IEO.Emb_df[self.IEO.Emb_df["Emb"]==emb].Data.values[0].split("%") for emb in list(self.IEO.Emb_df.Emb)}
            self.IEO.embedding_choice.options = self.IEO.Emb_list 
            self.IEO.embedding_choice.value = self.IEO.Emb_list[0]
            self.IEO.embedding_choice_out.clear_output()
            display(self.IEO.embedding_choice)
                
    def unlink_embedding(self, embedding_unlink_button):

        emb = self.embedding_select.value
        data = [ eml.split(" ")[0] if("\u2714" in eml) else eml for eml in self.embedding_data_select_multiple.value]
        
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
            

        with self.embedding_data_select_multiple_output:

            self.embedding_data_select_multiple_output.clear_output()

            for d in data:
                self.data_multi_select_dict_show[d] = d

            self.embedding_data_select_multiple.options = [self.data_multi_select_dict_show[dl] for dl in self.data_list]
            display(self.embedding_data_select_multiple)

        with self.IEO.text_dataset_out:
            
            self.IEO.text_dataset_out.clear_output()
            self.IEO.text_dataset.options =  [''] if(not(self.IEO.Emb_list[0] in self.IEO.Emb_dict.keys())) else self.IEO.Emb_dict[self.IEO.Emb_list[0]] + ['']
            self.IEO.text_dataset.value = ''
            display(self.IEO.text_dataset)

        with self.IEO.embedding_choice_out:
            
            self.IEO.Emb_list = list(pd.read_csv("Emb.csv").Emb)
            self.IEO.Emb_df = pd.read_csv("Emb#Data.csv") 
            self.IEO.Emb_dict = { emb: self.IEO.Emb_df[self.IEO.Emb_df["Emb"]==emb].Data.values[0].split("%") for emb in list(self.IEO.Emb_df.Emb)}
            self.IEO.embedding_choice.options = self.IEO.Emb_list 
            self.IEO.embedding_choice.value = self.IEO.Emb_list[0]
            self.IEO.embedding_choice_out.clear_output()
            display(self.IEO.embedding_choice)

        
            

