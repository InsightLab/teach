import ipywidgets as widgets
from IPython.display import display, HTML

import pandas as pd

import osmnx as ox
import networkx as nx

from time import sleep
from tqdm import trange
import nltk

import pathlib
import sys
from os import listdir

from copy import copy

import matplotlib.pyplot as plt

from pymove import filters
from pymove.visualization import folium as f
from pymove import MoveDataFrame
from pymove.visualization.folium import plot_trajectories, plot_points
from pymove.utils import distances



sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from utils.metrics import mean_reciprocal_rank_filter
from utils.distances import *
from utils.output import *

class Intrinsic_Evaluation_Output:

    def __init__(self, intrinsic_evaluation_output):

        self.embedding = "" 
        self.tokenizer_df = ""

        self.intrinsic_evaluation_output = intrinsic_evaluation_output

        self.Emb_list = list(pd.read_csv("Emb.csv").Emb)
        self.Emb_df = pd.read_csv("Emb#Data.csv") 
        self.Emb_dict = { emb: self.Emb_df[self.Emb_df["Emb"]==emb].Data.values[0].split("%") for emb in list(self.Emb_df.Emb)}


        self.text_dataset = widgets.Select(description="Select Data:",layout=widgets.Layout(width="380px",height="180px"), 
                                        options= [''] if(not(self.Emb_list[0] in self.Emb_dict.keys())) else self.Emb_dict[self.Emb_list[0]] + [''],value='')

        self.text_dataset.observe(self.on_change_traj_data,names="value")

        # Screen to select the dataset (lat_lon) linked to the chosen embedding
        self.text_dataset_out = out(self.text_dataset)

        self.embedding_text = widgets.Label("Select Embedding")
        self.embedding_choice = widgets.Dropdown(description='',layout=widgets.Layout(width="150px"),options=self.Emb_list,value=self.Emb_list[0])
        self.embedding_choice.observe(self.on_change_embedding_choice, names="value")

        # Dropdown used to choose the embedding
        self.embedding_choice_out = out(self.embedding_choice)

        self.spc = widgets.Label("")
        self.spc1 = widgets.Label("",layout=widgets.Layout(width="9%"))
        
        self.traj_data = ""
        self.t_id_list = []
        self.traj_id_dict_all = {}
        self.traj_id_dict = {}

        self.LatLong = ""
        self.LL = ""
        
        self.Dataset = ""
        self.embedding_matrix = ""
        self.Dataset_geo25 = ""
        self.Dict_Geo25 = {}

        self.Cossine_Matrix_sensors = ""
        self.CM_sensors = ""
        self.Eucli_Matrix_sensors = ""
        self.Road_Matrix_sensors = ""

        self.Cossine_Matrix= ""
        self.CM = ""
        self.DTW_Matrix = ""
        self.Edit_Matrix = ""
        
        self.Objects_traj = []
        self.obj_traj = ""
        self.Objects_loc = []
        self.obj_loc = ""
        
        self.query_index = ""
        
        self.Traj_Number_Traj_id = ""
        self.Dataset_geo25 = ""
        

        self.sensors_traj = []
        self.sts_id = []

        self.topk_loc = -1
        self.topk_traj = -1

        self.validation = True
        self.validation_2 = True
        #self.without_road_matrix = False

        
        # If there is no link between embedding and datasets of type lat_long

        if(len(list(self.Emb_df.Emb))==0):
            
            self.embedding_not_linked()
               
        # If there is link between embedding and datasets of type lat_long  
             
        elif(len(list(self.Emb_df.Emb))>1):
        
            self.embedding_linked()
            
            self.traj_matrix_first_read()
            
            # If searching for trajectory distance matrices yielded nothing, then creation of the matrices will is enabled

            if(self.validation):
            
                self.traj_matrix_first_creation()

            else:

                self.validation = True


            self.sensor_matrix_first_read()

            # If searching for location distance matrices yielded nothing, then creation of the matrices will is enabled
            
            if(self.validation_2):
                
                self.sensor_matrix_first_creation()

            else:

                self.validation_2 = True

        
        self.out_mrr_traj = out(widgets.Label(""))
        self.out_mrr_traj.layout = widgets.Layout(width = "450px", border='solid 2.0px white', margin='0px 5px 5px 5px', padding='2px 2px 2px 2px')
        
        self.out_mrr_loc = out(widgets.Label(""))
        self.out_mrr_loc.layout = widgets.Layout(width = "450px", border='solid 2.0px white', margin='0px 5px 5px 5px', padding='2px 2px 2px 2px')
        
        self.out_topk_traj = out(widgets.Label(""))
        self.out_topk_traj.layout = widgets.Layout(width = "825px", border='solid 2.0px white', margin='0px 10px 10px 10px', padding='5px 5px 5px 5px')
        
        self.out_topk_loc = out(widgets.Label(""))
        self.out_topk_loc.layout = widgets.Layout(width = "825px", border='solid 2.0px white', margin='0px 10px 10px 10px', padding='5px 5px 5px 5px')
        
        
        
        self.select_trajectory_label = widgets.Label("Select Trajectory")
        self.select_trajectory_dropdown = widgets.Dropdown(description='',options=self.Objects_traj,layout=widgets.Layout(width="150px"))
        self.select_trajectory_dropdown.observe(self.on_change_object_traj,names="value")

        # Trajectory dropdown for top-k plot
        self.select_trajectory_dropdown_output = out(self.select_trajectory_dropdown)
        
        self.select_location_label = widgets.Label("Select Loacation")
        self.select_location_dropdown = widgets.Dropdown(description='',options=self.Objects_loc,layout=widgets.Layout(width="150px"))
        self.select_location_dropdown.observe(self.on_change_object_loc,names="value")

        # Location dropdown for top-k plot
        self.select_location_dropdown_output = out(self.select_location_dropdown)


        # Number of similar trajectories
        self.top_k_trajectory_label = widgets.Label("Top-k")
        self.top_k_trajectory_text = widgets.Text(description='',layout=widgets.Layout(width="60px"))
        self.top_k_trajectory_text.observe(self.on_change_topk_traj,names="value")
        
        # Number of similar location
        self.top_k_location_label = widgets.Label("Top-k")
        self.top_k_location_text = widgets.Text(description='',layout=widgets.Layout(width="60px"))
        self.top_k_location_text.observe(self.on_change_topk_loc,names="value")

        # Button to plot the k similar trajectories
        self.top_k_trajectory_button = widgets.Button(description="Plot", layout= widgets.Layout(width="60px"))
        self.top_k_trajectory_button.on_click(self.traj_topk_plot)
        self.top_k_trajectory_button.style.button_color = "lightgray"
        
        
        # Button to plot the k similar locations
        self.top_k_location_button = widgets.Button(description="Plot", layout= widgets.Layout(width="60px"))
        self.top_k_location_button.on_click(self.loc_topk_plot)
        self.top_k_location_button.style.button_color = "lightgray"
        
        # Button to plot the mrr to the trajectories
        self.mrr_trajectory_plot_button = widgets.Button(description="Plot", layout= widgets.Layout(width="60px"))
        self.mrr_trajectory_plot_button.on_click(self.traj_plot_mrr)
        self.mrr_trajectory_plot_button.style.button_color = "lightgray"
        
        # Button to plot the mrr to the locations
        mrr_location_plot_button = widgets.Button(description="Plot", layout=widgets.Layout(width="60px")) 
        mrr_location_plot_button.on_click(self.loc_mrr_plot)
        mrr_location_plot_button.style.button_color = "lightgray"
        
        self.mrr_trajectory_plot_box = widgets.VBox([self.spc, widgets.HBox([self.spc,self.spc,self.spc,widgets.HBox([widgets.Label("MRR"),self.spc,self.mrr_trajectory_plot_button])]),widgets.VBox([self.spc,self.spc,self.out_mrr_traj])])
        
        self.mrr_location_plot_box = widgets.VBox([self.spc, widgets.HBox([self.spc,self.spc,self.spc,widgets.HBox([widgets.Label("MRR"),self.spc,mrr_location_plot_button])]),widgets.VBox([self.spc,self.spc,self.out_mrr_loc])])
        
        self.similar_trajectory_box = widgets.HBox([widgets.VBox([widgets.HBox([widgets.HBox([self.select_trajectory_label, self.spc, self.select_trajectory_dropdown_output]),self.spc, 
        widgets.HBox([self.top_k_trajectory_label,self.spc,self.top_k_trajectory_text,self.spc,self.spc,self.top_k_trajectory_button])]), widgets.HBox([self.spc,self.spc,self.spc,widgets.VBox([self.spc,self.out_topk_traj])])])])
        
        self.similar_location_box = widgets.VBox([widgets.VBox([widgets.HBox([widgets.HBox([self.select_location_label, self.spc, self.select_location_dropdown_output]),self.spc, 
        widgets.HBox([self.top_k_location_label,self.spc,self.top_k_location_text,self.spc,self.spc,self.top_k_location_button])]),widgets.HBox([self.spc,self.spc,self.spc,widgets.VBox([self.spc,self.out_topk_loc])])])])
        
        self.intrinsic_accordion = widgets.Accordion(children=[self.mrr_location_plot_box,self.similar_location_box,self.mrr_trajectory_plot_box,self.similar_trajectory_box],selected_index=None)
        
        self.intrinsic_accordion.set_title(0, 'Mean Reciprocal Rank (MRR) Location')
        self.intrinsic_accordion.set_title(1, 'Similar Locations')
        self.intrinsic_accordion.set_title(2, 'Mean Reciprocal Rank (MRR) Trajectory')
        self.intrinsic_accordion.set_title(3, 'Similar Trajectories')
        

        self.intrinsic_evaluation_box = widgets.VBox([widgets.HBox([widgets.VBox([self.embedding_text,self.embedding_choice_out]),self.spc,self.spc,self.spc1, 
                                                      widgets.VBox([self.text_dataset_out])]),self.spc, self.intrinsic_accordion])
        
    #def road_matrix_yes(self):

    #    self.road_matrix_creation()

    #def road_matrix_no(self):

    #    self.without_road_matrix = True
    #    self.intrinsic_evaluation_output.clear_output()

    def on_change_object_traj(self,change):

        self.obj_traj = change.new

    def on_change_object_loc(self, change):

        self.obj_loc = change.new

    def on_change_topk_loc(self, change):

        self.topk_loc = change.new

    def on_change_topk_traj(self, change):

        self.topk_traj = change.new
        
    def on_change_embedding_choice(self, change):
            
        self.embedding = change.new
            
        aux_emb = pd.read_csv("embeddings/" + self.embedding + ".csv") 
    
        self.tokenizer_df = aux_emb.loc[0:list(pd.isnull(aux_emb["sensor"])).index(True)-1,["sensor","id"]]
        self.tokenizer_df.id = [int(i) for i in self.tokenizer_df.id]
        self.tokenizer_df.index = self.tokenizer_df.id
        self.tokenizer_df = self.tokenizer_df[["sensor","id"]]

        self.embeddding_matrix = aux_emb.loc[list(pd.isnull(aux_emb["sensor"])).index(True):int(list(pd.isnull(aux_emb["sensor"])).index(True)*2),
                        [str(i) for i in range(len(aux_emb.columns)-2)]]
        
        self.embeddding_matrix.index = [i for i in range(self.embeddding_matrix.shape[0])]

        # When choosing embedding , the tokenizer and embedding array are extracted from the file. After that, the screen of the datasets linked to the selected embedding is modified.
        
        with self.text_dataset_out:
            
            self.text_dataset_out.clear_output()
            
            display(widgets.Label("Processing..."))
        
            self.text_dataset.options = [''] if(not(self.embedding in self.Emb_dict.keys())) else self.Emb_dict[self.embedding] + ['']
            
            self.text_dataset.value = ''
            
            self.text_dataset_out.clear_output()
            
            display(self.text_dataset)


    def embedding_not_linked(self):

        # If the link does not exist, only the tokenizer and the embedding matrix are loaded.

        aux_emb = pd.read_csv("embeddings/" + self.Emb_list[0] + ".csv") 

        self.tokenizer_df = aux_emb.loc[0:list(pd.isnull(aux_emb["sensor"])).index(True)-1,["sensor","id"]]
        self.tokenizer_df.id = [int(i) for i in self.tokenizer_df.id]
        self.tokenizer_df.index = self.tokenizer_df.id
        self.tokenizer_df = self.tokenizer_df[["sensor","id"]]

        self.embedding_matrix = aux_emb.loc[list(pd.isnull(aux_emb["sensor"])).index(True):int(list(pd.isnull(aux_emb["sensor"])).index(True)*2),
                        [str(i) for i in range(len(aux_emb.columns)-2)]]

        self.embedding_matrix.index = [i for i in range(self.embedding_matrix.shape[0])]
        
        self.LL = pd.read_csv("lat_lon_sensors/"+"sensors_roubados_representativos_countmin18(1).csv", index_col='geos25')
        self.LL = self.LL.astype({'lat':'float32', 'lon':'float32'})
        self.LL.drop_duplicates(inplace=True)
        self.LL['sensor'] = [str(geos25).lower() for geos25 in self.LL.index.values]
        self.LL.set_index('sensor', inplace=True)

    def embedding_linked(self):

        aux_emb = pd.read_csv("embeddings/" + self.Emb_list[0] + ".csv") 

        self.tokenizer_df = aux_emb.loc[0:list(pd.isnull(aux_emb["sensor"])).index(True)-1,["sensor","id"]]
        self.tokenizer_df.id = [int(i) for i in self.tokenizer_df.id]
        self.tokenizer_df.index = self.tokenizer_df.id
        self.tokenizer_df = self.tokenizer_df[["sensor","id"]]

        self.embedding_matrix = aux_emb.loc[list(pd.isnull(aux_emb["sensor"])).index(True):int(list(pd.isnull(aux_emb["sensor"])).index(True)*2), [str(i) for i in range(len(aux_emb.columns)-2)]]

        self.embedding_matrix.index = [i for i in range(self.embedding_matrix.shape[0])]
        
        self.traj_data = self.Emb_dict[self.Emb_list[0]][0]
        
        self.embedding = self.Emb_list[0]
        
        self.LatLong   = pd.read_csv("data/"+self.traj_data + ".csv") 
        aux_latlon = sorted(list(set(self.LatLong["trajectory_id"])))
        
        self.LL = pd.read_csv("lat_lon_sensors/"+"sensors_roubados_representativos_countmin18(1).csv", index_col='geos25')
        self.LL = self.LL.astype({'lat':'float32', 'lon':'float32'})
        self.LL.drop_duplicates(inplace=True)
        self.LL['sensor'] = [str(geos25).lower() for geos25 in self.LL.index.values]
        self.LL.set_index('sensor', inplace=True)
        
        self.t_id_list = []
            
        for tid in aux_latlon:

            self.t_id_list.append(tid)
                
        
        self.t_id_list = sorted(self.t_id_list)
        self.traj_id_dict_all = {self.t_id_list[i]: i for i in range(len(self.t_id_list))}


        # As there is a link between the incorporation and the dataset, it is checked if there are already datasets of type sequence, if not, it is created.

        trajs = listdir("./trajectories")

        if(self.embedding + "_" + self.traj_data + "_trajs" + ".csv" in trajs):

            self.LatLong = pd.read_csv("data/" + self.traj_data + ".csv") 
            t_id_list = sorted(list(set(self.LatLong["trajectory_id"])))
            self.traj_id_dict = {tid: self.traj_id_dict_all[tid] for tid in t_id_list}
            self.Objects_traj = sorted(list(self.traj_id_dict.values()))
            self.obj_traj = self.Objects_traj[0]

            trajs_token = sorted(self.traj_id_dict.keys())

            self.Dataset = pd.read_csv("trajectories/" + self.embedding + "_" + self.traj_data + "_trajs" + ".csv") 
            
            self.Dataset["0"] = self.Dataset["0"].apply(lambda x : x.replace("[","").replace("]","").replace(",","").replace("\n","").replace("'",""))
            self.Dataset["0"] = self.Dataset["0"].apply(lambda x : x.split())
            self.Dataset["trajectory_number"] = self.Dataset["trajectory_number"].apply(lambda x: str(x))

            self.Traj_Number_Traj_id = pd.DataFrame({"0":trajs_token},index=[str(self.traj_id_dict[i]) for i in trajs_token])

            self.Dataset_geo25 = copy(self.Dataset)

            self.Dict_Geo25 = { geo25.upper():vector  for geo25,vector in zip(list(self.tokenizer_df['sensor'].values),np.array(self.embedding_matrix))}


            self.Dataset["0"] = self.Dataset["0"].apply(lambda x:[ self.Dict_Geo25[geo25] if(geo25 in self.Dict_Geo25.keys()) else np.zeros((1,self.embedding_matrix.shape[1]))[0] for geo25 in x ]) 
            self.Dataset["0"] = self.Dataset["0"].apply(lambda x : np.array(sum(x)/len(x)))

        else:

            self.LatLong = pd.read_csv("data/" + self.traj_data + ".csv") 
            t_id_list = sorted(list(set(self.LatLong["trajectory_id"])))
            self.traj_id_dict = {tid: self.traj_id_dict_all[tid] for tid in t_id_list}
            self.Objects_traj = sorted(list(self.traj_id_dict.values()))
            self.obj_traj = self.Objects_traj[0]
            trajs_token = sorted(self.traj_id_dict.keys())

            traj_dict = {"0":[list(self.LatLong[self.LatLong["trajectory_id"] == traj]["location_label"]) for traj in trajs_token],"trajectory_number":[str(self.traj_id_dict[i]) for i in trajs_token]}
            self.Dataset = pd.DataFrame(traj_dict,index=[str(self.traj_id_dict[i]) for i in trajs_token])
            self.Dataset.to_csv("trajectories/"+ self.embedding + "_" + self.traj_data + "_trajs" + ".csv",index=False)

            self.Dataset["trajectory_number"] = self.Dataset["trajectory_number"].apply(lambda x: str(x))

            self.Traj_Number_Traj_id = pd.DataFrame({"0":trajs_token},index=[str(self.traj_id_dict[i]) for i in trajs_token])

            self.Dataset_geo25 = copy(self.Dataset)

            self.Dict_Geo25 = { geo25.upper():vector  for geo25,vector in zip(list(self.tokenizer_df['sensor'].values),np.array(self.embedding_matrix))}


            self.Dataset["0"] = self.Dataset["0"].apply(lambda x:[ self.Dict_Geo25[geo25] if(geo25 in self.Dict_Geo25.keys()) else np.zeros((1,self.embedding_matrix.shape[1]))[0] for geo25 in x ]) 
            self.Dataset["0"] = self.Dataset["0"].apply(lambda x : np.array(sum(x)/len(x)))




    def traj_matrix_first_read(self):

        # Initially, it checks if there are distance matrices for trajectories associated with the embedding and its dataset of type lat_lon that was chosen, among those that were linked.

        matrice = listdir("./matrices")

        if(self.embedding + "_" + self.traj_data  +"_cossine_matrix_trajs.csv" in matrice):

            self.Cossine_Matrix = pd.read_csv("matrices/"  + self.embedding + "_" + self.traj_data  +"_cossine_matrix_trajs.csv")   
            self.CM = pd.read_csv("matrices/"  + self.embedding + "_" + self.traj_data + "_cm_trajs.csv")
            self.DTW_Matrix = pd.read_csv("matrices/"  + self.embedding + "_" + self.traj_data + "_dtw_matrix_trajs.csv")           
            self.Edit_Matrix = pd.read_csv("matrices/" +  self.embedding + "_" + self.traj_data + "_edit_matrix_trajs.csv") 

            self.query_index = np.arange(0,self.Cossine_Matrix.shape[0])

            self.sts_id = []
            
            for index in self.tokenizer_df.index:
                self.sts_id.append(index)

            self.validation = False

                
    def traj_matrix_first_creation(self):

        self.Dataset.index = [i for i in range(len(self.Dataset))]
         
        dist_matrix_coss_traj = np.zeros((self.Dataset["0"].shape[0],self.Dataset["0"].shape[0]))

        for i in range(self.Dataset["0"].shape[0]):
            for j in range(self.Dataset["0"].shape[0]):
                dist_matrix_coss_traj[i][j] = nltk.cluster.cosine_distance(self.Dataset["0"][i],self.Dataset["0"][j])


        self.Cossine_Matrix = pd.DataFrame(dist_matrix_coss_traj)

        self.query_index = np.arange(0,self.Cossine_Matrix.shape[0])

        self.CM = copy(self.Cossine_Matrix)
        self.CM.to_csv("matrices/" + self.embedding + "_" + self.traj_data +"_cm_trajs.csv",index=False)
        
        for i in range(0,self.Cossine_Matrix.shape[0]):

            self.Cossine_Matrix.iloc[i,i] = np.nan 
            
        self.Cossine_Matrix.to_csv("matrices/"  + self.embedding + "_" +self.traj_data + "_cossine_matrix_trajs.csv",index=False)
        
        
        matrix_dist_dtw = np.zeros((len(self.t_id_list),len(self.t_id_list)), dtype='float16')

        TJ_ID = self.t_id_list

        for i in range(len(TJ_ID)):
            for j in range(i,len(TJ_ID)):
                matrix_dist_dtw[i][j] = dtw_distance(TJ_ID[i], TJ_ID[j], self.LatLong)
                
        for i in reversed(range(len(TJ_ID))):
            for j in reversed(range(i)):
                matrix_dist_dtw[i][j] = matrix_dist_dtw[j][i]
                
        self.DTW_Matrix = pd.DataFrame(matrix_dist_dtw)


        dist_matrix = -np.ones((self.Dataset_geo25.shape[0],self.Dataset_geo25.shape[0]), dtype='float16')

        ids_list = list(self.Dataset_geo25["0"].index)
        for i in range(len(ids_list)):
            for j in range(len(ids_list)):
                dist_matrix[i][j] = edit_distance(self.Dataset_geo25["0"].loc[ids_list[i]],self.Dataset_geo25["0"].loc[ids_list[j]])
                
        self.Edit_Matrix = pd.DataFrame(dist_matrix)
        
        for i in range(0,self.DTW_Matrix.shape[0]):

            self.DTW_Matrix.iloc[i,i] = np.nan
            self.Edit_Matrix.iloc[i,i] = np.nan
            
            
        self.DTW_Matrix.to_csv("matrices/"  + self.embedding + "_" + self.traj_data +"_dtw_matrix_trajs.csv",index=False)
        self.Edit_Matrix.to_csv("matrices/" + self.embedding + "_" + self.traj_data +"_edit_matrix_trajs.csv",index=False)
        

        self.sts_id = []

        for index in self.tokenizer_df.index:
            self.sts_id.append(index)

      
    def sensor_matrix_first_read(self):

        # At first it is checked if the Road_Matrix exists, because it is only read if it exists, otherwise it is created. After that your reading is performed.

        if("Road_Matrix_sensors.csv" in listdir("./matrices")):

            mat = listdir("./matrices")

            # In the two conditions above, a search is made in the matrices folder for the cosine matrix associated with the embedding

            if(self.embedding + "_cossine_matrix_sensors.csv" in mat):
        
                self.Road_Matrix_sensors = pd.read_csv("matrices/Road_Matrix_sensors.csv")
                self.Road_Matrix_sensors.index = [int(i) for i in self.Road_Matrix_sensors.columns]

                self.Cossine_Matrix_sensors = pd.read_csv("matrices/" + self.embedding + "_cossine_matrix_sensors.csv")
                self.CM_sensors = pd.read_csv("matrices/" + self.embedding + "_cm_sensors.csv")
                self.Eucli_Matrix_sensors = pd.read_csv("matrices/" + self.embedding + "_euclidean_matrix_sensors.csv")
                
                self.Cossine_Matrix_sensors.index = [int(i) for i in self.Cossine_Matrix_sensors.index]
                self.CM_sensors.index = [int(i) for i in self.CM_sensors.index]
                self.CM_sensors.columns = [int(i) for i in self.CM_sensors.columns]
                self.Eucli_Matrix_sensors.index = [int(i) for i in self.Eucli_Matrix_sensors.index]
                
                #if(not(self.without_road_matrix)):
                columns_sensors = np.array(sorted(np.intersect1d(self.Eucli_Matrix_sensors.columns.values,np.intersect1d(self.Cossine_Matrix_sensors.columns.values, self.Road_Matrix_sensors.columns.values)),key=lambda x : int(x)))
                index_sensors = np.intersect1d(self.Eucli_Matrix_sensors.index.values,np.intersect1d(self.Cossine_Matrix_sensors.index.values, self.Road_Matrix_sensors.index.values))
                self.Road_Matrix_sensors = self.Road_Matrix_sensors.loc[index_sensors,columns_sensors]
                
                #self.LatLong["trajectory_id"] = self.LatLong.index
                
                if(101 in self.sts_id):
                    self.sts_id.remove(101)
                if(102 in self.sts_id):
                    self.sts_id.remove(102)
                if(103 in self.sts_id):
                    self.sts_id.remove(103)

                self.Objects_loc = self.sts_id
                self.obj_loc = self.Objects_loc[0]
                
                self.validation_2 = False
            
        else:

            if(isinstance(self.Road_Matrix_sensors,str)):
                if("Road_Matrix_sensors.csv" in listdir("./matrices")):
                    self.Road_Matrix_sensors = pd.read_csv("matrices/Road_Matrix_sensors.csv")
                else:

                    self.road_matrix_creation()
            #with self.intrinsic_evaluation_output:

            #    self.intrinsic_evaluation_output.clear_output()

            #    road_label = widgets.Label("The Road Matrix is ​​not in a folder, would you like to generate it (it may take a long time)?")

            #    yes_button = widgets.Button(description="Yes", layout=widgets.Layout(width="100px"))
            #    yes_button.style.button_color = "lightgray"
            #    yes_button.on_click(self.road_matrix_yes)

            #    no_button = widgets.Button(description="No", layout=widgets.Layout(width="100px"))
            #    no_button.style.button_color = "lightgray"
            #    no_button.on_click(self.road_matrix_no)

            #    yes_no_box =  widgets.HBox([yes_button, self.spc1, no_button])

            #    road_box = widgets.VBox([road_label, self.spc1, yes_no_box])

            #    display(road_box)

            mat = listdir("./matrices")

            if(self.embedding + "_cossine_matrix_sensors.csv" in mat):
            
                
                self.Cossine_Matrix_sensors = pd.read_csv("matrices/" + self.embedding + "_cossine_matrix_sensors.csv")
                self.CM_sensors = pd.read_csv("matrices/" + self.embedding + "_cm_sensors.csv")
                self.Eucli_Matrix_sensors = pd.read_csv("matrices/" + self.embedding + "_euclidean_matrix_sensors.csv")
                
                self.Cossine_Matrix_sensors.index = [int(i) for i in self.Cossine_Matrix_sensors.index]
                self.CM_sensors.index = [int(i) for i in self.CM_sensors.index]
                self.CM_sensors.columns = [int(i) for i in self.CM_sensors.columns]
                self.Eucli_Matrix_sensors.index = [int(i) for i in self.Eucli_Matrix_sensors.index]
                
                #if(not(self.without_road_matrix)):
                columns_sensors = np.array(sorted(np.intersect1d(self.Eucli_Matrix_sensors.columns.values,np.intersect1d(self.Cossine_Matrix_sensors.columns.values, self.Road_Matrix_sensors.columns.values)),key=lambda x : int(x)))
                index_sensors = np.intersect1d(self.Eucli_Matrix_sensors.index.values,np.intersect1d(self.Cossine_Matrix_sensors.index.values, self.Road_Matrix_sensors.index.values))
                self.Road_Matrix_sensors = self.Road_Matrix_sensors.loc[index_sensors,columns_sensors]
                
                #self.LatLong["trajectory_id"] = self.LatLong.index
                
                if(101 in self.sts_id):
                    self.sts_id.remove(101)
                if(102 in self.sts_id):
                    self.sts_id.remove(102)
                if(103 in self.sts_id):
                    self.sts_id.remove(103)

                self.Objects_loc = self.sts_id
                self.obj_loc = self.Objects_loc[0]
                
                self.validation_2 = False


    def sensor_matrix_first_creation(self):

        
        if(isinstance(self.Road_Matrix_sensors,str)):
            if("Road_Matrix_sensors.csv" in listdir("./matrices")):
                self.Road_Matrix_sensors = pd.read_csv("matrices/Road_Matrix_sensors.csv")
            else:

                self.road_matrix_creation()
                #with self.intrinsic_evaluation_output:

                #    self.intrinsic_evaluation_output.clear_output()

                #    road_label = widgets.Label("The Road Matrix is ​​not in a folder, would you like to generate it (it may take a long time)?")

                #    yes_button = widgets.Button(description="Yes", layout=widgets.Layout(width="100px"))
                #    yes_button.style.button_color = "lightgray"
                #    yes_button.on_click(self.road_matrix_yes)

                #    no_button = widgets.Button(description="No", layout=widgets.Layout(width="100px"))
                #    no_button.style.button_color = "lightgray"
                #    no_button.on_click(self.road_matrix_no)

                #    yes_no_box =  widgets.HBox([yes_button, self.spc1, no_button])

                #    road_box = widgets.VBox([road_label, self.spc1, yes_no_box])

                #    display(road_box)

        dist_matrix_coss_sensor = np.zeros((len(self.sts_id),len(self.sts_id)))
        eucl_distance = np.zeros((len(self.sts_id),len(self.sts_id)))

        for i in range(len(self.sts_id)):
            for j in range(len(self.sts_id)):
                dist_matrix_coss_sensor[i,j] = nltk.cluster.cosine_distance(np.array(self.embedding_matrix)[self.sts_id[i]-1],np.array(self.embedding_matrix)[self.sts_id[j]-1])


        lat_values = []


        for i in self.tokenizer_df.index:

            if(i not in [101,102,103] ):
                lat_values.append(self.LL.loc[self.tokenizer_df[self.tokenizer_df["id"] == i]["sensor"].apply(lambda x : x.lower())]["lat"].values[0])

            else:
                lat_values.append("n")

        lon_values = []

        for i in self.tokenizer_df.index:

            if(i not in [101,102,103]):
                lon_values.append(self.LL.loc[self.tokenizer_df[self.tokenizer_df["id"] == i]["sensor"].apply(lambda x : x.lower())]["lon"].values[0])

            else:
                lon_values.append("n")    


        for i in range(len(lat_values)):
            for j in range(len(lat_values)):
                if((lat_values[i] != "n") and (lat_values[j] != "n") ):     
                    eucl_distance[i,j] = distances.euclidean_distance_in_meters(lat_values[i],lon_values[i],lat_values[j],lon_values[j])



        self.Cossine_Matrix_sensors = pd.DataFrame(dist_matrix_coss_sensor)

        self.CM_sensors = copy(self.Cossine_Matrix_sensors)
        self.CM_sensors.to_csv("matrices/" + self.embedding + "_cm_sensors.csv",index=False)

        self.Eucli_Matrix_sensors = pd.DataFrame(eucl_distance)

        for i in self.Cossine_Matrix_sensors.index:

            self.Cossine_Matrix_sensors.iloc[i,i] = np.nan 

        for i in self.Eucli_Matrix_sensors.index:

            self.Eucli_Matrix_sensors.iloc[i,i] = np.nan

        self.CM_sensors= copy(self.Cossine_Matrix_sensors)
        self.CM_sensors.columns = [int(i) for i in self.CM_sensors.columns]
        self.CM_sensors.to_csv("matrices/" + self.embedding + "_cm_sensors.csv",index=False)

        self.Cossine_Matrix_sensors.columns = [str(i) for i in self.sts_id]
        self.Eucli_Matrix_sensors.columns = [str(i) for i in self.sts_id]
        self.Cossine_Matrix_sensors.index = self.sts_id
        self.Eucli_Matrix_sensors.index = self.sts_id

        #if(not(self.without_road_matrix)):
        columns_sensors = np.array(sorted(np.intersect1d(self.Eucli_Matrix_sensors.columns.values,np.intersect1d(self.Cossine_Matrix_sensors.columns.values, self.Road_Matrix_sensors.columns.values)),key=lambda x : int(x)))
        index_sensors = np.intersect1d(self.Eucli_Matrix_sensors.index.values,np.intersect1d(self.Cossine_Matrix_sensors.index.values, self.Road_Matrix_sensors.index.values))
        
        self.Cossine_Matrix_sensors = self.Cossine_Matrix_sensors.loc[index_sensors,columns_sensors]
        self.Eucli_Matrix_sensors = self.Eucli_Matrix_sensors.loc[index_sensors,columns_sensors]
        self.Road_Matrix_sensors = self.Road_Matrix_sensors.loc[index_sensors,columns_sensors]

        self.Cossine_Matrix_sensors.to_csv("matrices/" + self.embedding + "_cossine_matrix_sensors.csv",index=False)
        self.Eucli_Matrix_sensors.to_csv("matrices/" + self.embedding + "_euclidean_matrix_sensors.csv",index=False)
    

        #self.LatLong["trajectory_id"] =  self.LatLong.index
        
        if(101 in self.sts_id):
            self.sts_id.remove(101)
        if(102 in self.sts_id):
            self.sts_id.remove(102)
        if(103 in self.sts_id):
            self.sts_id.remove(103)
        
        self.Objects_loc = self.sts_id
        self.obj_loc = self.Objects_loc[0]


    def road_matrix_creation(self):

        # A screen with a progress bar is created, and it is possible to see the progress of the creation of the Road_Matrix.
        with self.intrinsic_evaluation_output:

            self.intrinsic_evaluation_output.clear_output()
            display(widgets.Label("Generating Road Matrix ..."))
            
            id_road = [] 
            sensors = list(self.tokenizer_df.loc[self.sts_id]["sensor"].apply(lambda x : x.upper()).values)


            for index in self.LL.index:
                if(index.upper() in sensors):
                    id_road.append(index)

            lat_long = self.LL.loc[id_road]

            south, west = lat_long[['lat','lon']].min()
            north, east = lat_long[['lat','lon']].max()

            g = ox.graph_from_bbox(north, south, east, west, network_type='drive_service')

            id_remove = []
            for index in lat_long.index:
                if(not(lat_long.loc[index].nodeId in g.nodes)):      
                    id_remove.append(index)

            lat_long = lat_long.drop(id_remove)

            id_w_duplicate = []
            node_id_w_duplicate = []

            for index in lat_long.index:
                if(not(lat_long.loc[index].nodeId in node_id_w_duplicate)):      
                    id_w_duplicate.append(index)
                    node_id_w_duplicate.append(lat_long.loc[index].nodeId)

            lat_long = lat_long.loc[id_w_duplicate]

            lat_long = lat_long.sort_values(by=['token_id'])

            net_dist = np.zeros((len(lat_long.index.values),len(lat_long.index.values)))
            node_list = [lat_long.loc[lat_long.index[s]]['nodeId'] for s in range(len(lat_long.index))]

            results = []

            pairs = [(g,u,v) for u in node_list for v in node_list if u != v]


            for i in trange(len(pairs)):

                try:
                    results.append(nx.shortest_path_length(pairs[i][0],pairs[i][1],pairs[i][2],weight='length'))

                except:

                    results.append(np.nan)

                sleep(0.01)


            for i in range(len(pairs)):

                x = node_list.index(pairs[i][1]) 
                y = node_list.index(pairs[i][2])

                net_dist[x][y] = results[i]

            for i in range(len(net_dist)):
                net_dist[i][i] = np.nan

            df_dist = pd.DataFrame(net_dist)
            df_dist.index = list(lat_long.token_id)
            df_dist.columns = list(lat_long.token_id)

            self.Road_Matrix_sensors = df_dist
            self.Road_Matrix_sensors.columns = [str(i) for i in self.Road_Matrix_sensors.columns]
            self.Road_Matrix_sensors.to_csv("matrices/" + "Road_Matrix_sensors.csv",index=False)
            
            self.intrinsic_evaluation_output.clear_output()
            display(widgets.Label("Processing..."))

    def data_traj_list_dict(self):
        
        self.LatLong   = pd.read_csv("data/"+self.traj_data+".csv") 
        self.t_id_list = []


        for tid in list(set(self.LatLong["trajectory_id"])):

            self.t_id_list.append(tid)

        self.t_id_list = sorted(self.t_id_list)
        self.traj_id_dict_all = {self.t_id_list[i]: i for i in range(len(self.t_id_list))}




    def traj_matrix_read(self):    

        trajs = listdir("./trajectories")

        # It is checked if there is any dataset of type sequence relating the embedding and the dataset of type lat_lon in the trajectories folder.

        if(self.embedding + "_" + self.traj_data + "_trajs" + ".csv" in trajs):                                                                                                                                                                                                                                                                                   

            self.LatLong = pd.read_csv("data/" + self.traj_data + ".csv") 
            t_id_list = sorted(list(set(self.LatLong["trajectory_id"])))
            self.traj_id_dict = {tid: self.traj_id_dict_all[tid] for tid in t_id_list}

            self.Objects_traj = sorted(list(self.traj_id_dict.values()))
            self.obj_traj = self.Objects_traj[0]

            with self.select_location_dropdown_output:

                self.select_trajectory_dropdown.options = self.Objects_traj
                self.select_trajectory_dropdown.value = self.obj_traj

            trajs_token = sorted(self.traj_id_dict.keys())

            self.Dataset = pd.read_csv("trajectories/" + self.embedding + "_" + self.traj_data + "_trajs" + ".csv") 

            self.Dataset["0"] = self.Dataset["0"].apply(lambda x : x.replace("[","").replace("]","").replace(",","").replace("\n","").replace("'",""))
            self.Dataset["0"] = self.Dataset["0"].apply(lambda x : x.split())
            self.Dataset["trajectory_number"] = self.Dataset["trajectory_number"].apply(lambda x: str(x))

            self.Traj_Number_Traj_id = pd.DataFrame({"0":trajs_token},index=[str(self.traj_id_dict[i]) for i in trajs_token])

            self.Dataset_geo25 = copy(self.Dataset)
            self.Dict_Geo25 = { geo25.upper():vector  for geo25,vector in zip(list(self.tokenizer_df['sensor'].values),np.array(self.embedding_matrix))}

            self.Dataset["0"] = self.Dataset["0"].apply(lambda x:[ self.Dict_Geo25[geo25] if(geo25 in self.Dict_Geo25.keys()) else np.zeros((1,self.embedding_matrix.shape[1]))[0] for geo25 in x ]) 
            self.Dataset["0"] = self.Dataset["0"].apply(lambda x : np.array(sum(x)/len(x)))


            self.Cossine_Matrix = pd.read_csv("matrices/" +  self.embedding + "_" + self.traj_data + "_cossine_matrix_trajs.csv")
            self.CM = pd.read_csv("matrices/" + self.embedding + "_" + self.traj_data + "_cm_trajs.csv")
            self.DTW_Matrix = pd.read_csv("matrices/" + self.embedding + "_" + self.traj_data + "_dtw_matrix_trajs.csv") 
            self.Edit_Matrix = pd.read_csv("matrices/" +  self.embedding + "_" + self.traj_data + "_edit_matrix_trajs.csv") 

            self.sts_id  = []

            for index in self.tokenizer_df.index:
                self.sts_id.append(index)
            
            self.validation = False
                             
    def traj_matrix_creation(self):

        trajs = listdir("./trajectories")

        # Initially, a reading is performed to find out if there is a dataset of the sequence type representing this link in the trajectories folder, if not, this dataset is created

        if(not(self.embedding + "_" + self.traj_data + "_trajs" + ".csv" in trajs)):

            self.LatLong = pd.read_csv("data/" + self.traj_data + ".csv") 
            t_id_list = sorted(list(set(self.LatLong["trajectory_id"])))
            self.traj_id_dict = {tid: self.traj_id_dict_all[tid] for tid in t_id_list}
            self.Objects_traj = sorted(list(self.traj_id_dict.values()))
            self.obj_traj = self.Objects_traj[0]
            trajs_token = sorted(self.traj_id_dict.keys())

            traj_dict = {"0":[list(self.LatLong[self.LatLong["trajectory_id"] == traj]["location_label"]) for traj in trajs_token],"trajectory_number":[str(self.traj_id_dict[i]) for i in trajs_token]}
            self.Dataset = pd.DataFrame(traj_dict,index=[str(self.traj_id_dict[i]) for i in trajs_token])
            self.Dataset.to_csv("trajectories/"+ self.embedding + "_" + self.traj_data + "_trajs"+".csv",index=False)




        self.Dataset = pd.read_csv("trajectories/" + self.embedding + "_" + self.traj_data + "_trajs" + ".csv") 
        self.Dataset["0"] = self.Dataset["0"].apply(lambda x : x.replace("[","").replace("]","").replace(",","").replace("\n","").replace("'",""))
        self.Dataset["0"] = self.Dataset["0"].apply(lambda x : x.split())
        self.Dataset["trajectory_number"] = self.Dataset["trajectory_number"].apply(lambda x: str(x))

        self.Traj_Number_Traj_id = pd.DataFrame({"0":trajs_token},index=[str(self.traj_id_dict[i]) for i in trajs_token])

        self.Dataset_geo25 = copy(self.Dataset)
        self.Dict_Geo25 = { geo25.upper():vector  for geo25,vector in zip(list(self.tokenizer_df['sensor'].values),np.array(self.embedding_matrix))}


        self.Dataset["0"] = self.Dataset["0"].apply(lambda x:[ self.Dict_Geo25[geo25] if(geo25 in self.Dict_Geo25.keys()) else np.zeros((1,self.embedding_matrix.shape[1]))[0] for geo25 in x ]) 
        self.Dataset["0"] = self.Dataset["0"].apply(lambda x : np.array(sum(x)/len(x)))

        self.Dataset.index = [i for i in range(len(self.Dataset))]
       
        dist_matrix_coss_traj = np.zeros((self.Dataset["0"].shape[0],self.Dataset["0"].shape[0]))

        for i in range(self.Dataset["0"].shape[0]):
            for j in range(self.Dataset["0"].shape[0]):
                dist_matrix_coss_traj[i][j] = nltk.cluster.cosine_distance(self.Dataset["0"][i],self.Dataset["0"][j])


        self.Cossine_Matrix = pd.DataFrame(dist_matrix_coss_traj)

        self.query_index = np.arange(0,self.Cossine_Matrix.shape[0])


        self.CM = copy(self.Cossine_Matrix)
        self.CM.to_csv("matrices/" + self.embedding + "_" + self.traj_data + "_cm_trajs.csv",index=False)

        for i in range(0,self.Cossine_Matrix.shape[0]):

            self.Cossine_Matrix.iloc[i,i] = np.nan 

        self.Cossine_Matrix.to_csv("matrices/" + self.embedding + "_" + self.traj_data + "_cossine_matrix_trajs.csv",index=False)


        matrix_dist_dtw = np.zeros((len(self.t_id_list),len(self.t_id_list)), dtype='float16')

        TJ_ID = self.t_id_list

        for i in range(len(TJ_ID)):
            for j in range(i,len(TJ_ID)):
                matrix_dist_dtw[i][j] = dtw_distance(TJ_ID[i], TJ_ID[j], self.LatLong)

        for i in reversed(range(len(TJ_ID))):
            for j in reversed(range(i)):
                matrix_dist_dtw[i][j] = matrix_dist_dtw[j][i]

        self.DTW_Matrix = pd.DataFrame(matrix_dist_dtw)


        dist_matrix = -np.ones((self.Dataset_geo25.shape[0],self.Dataset_geo25.shape[0]), dtype='float16')

        ids_list = list(self.Dataset_geo25["0"].index)
        for i in range(len(ids_list)):
            for j in range(len(ids_list)):
                dist_matrix[i][j] = edit_distance(self.Dataset_geo25["0"].loc[ids_list[i]],self.Dataset_geo25["0"].loc[ids_list[j]])

        self.Edit_Matrix = pd.DataFrame(dist_matrix)


        for i in range(0,self.DTW_Matrix.shape[0]):

            self.DTW_Matrix.iloc[i,i] = np.nan
            self.Edit_Matrix.iloc[i,i] = np.nan


        self.DTW_Matrix.to_csv("matrices/" + self.embedding + "_" + self.traj_data + "_dtw_matrix_trajs.csv",index=False)
        self.Edit_Matrix.to_csv("matrices/" +  self.embedding + "_" + self.traj_data + "_edit_matrix_trajs.csv",index=False)


        self.sts_id  = []
        for index in self.tokenizer_df.index:
            self.sts_id.append(index)


    def sensor_matrix_read(self):

        mat = listdir("./matrices")

        if(self.embedding + "_cossine_matrix_sensors.csv" in mat):

            if(isinstance(self.Road_Matrix_sensors,str)):
                if("Road_Matrix_sensors.csv" in listdir("./matrices")):
                    self.Road_Matrix_sensors = pd.read_csv("matrices/Road_Matrix_sensors.csv")
                else:

                    self.road_matrix_creation()
                    #with self.intrinsic_evaluation_output:

                    #self.intrinsic_evaluation_output.clear_output()

                    #road_label = widgets.Label("The Road Matrix is ​​not in a folder, would you like to generate it (it may take a long time)?")

                    #yes_button = widgets.Button(description="Yes", layout=widgets.Layout(width="100px"))
                    #yes_button.style.button_color = "lightgray"
                    #yes_button.on_click(self.road_matrix_yes)

                    #no_button = widgets.Button(description="No", layout=widgets.Layout(width="100px"))
                    #no_button.style.button_color = "lightgray"
                    #no_button.on_click(self.road_matrix_no)

                    #yes_no_box =  widgets.HBox([yes_button, self.spc1, no_button])

                    #road_box = widgets.VBox([road_label, self.spc1, yes_no_box])

                    #display(road_box)

            self.Cossine_Matrix_sensors = pd.read_csv("matrices/" + self.embedding + "_cossine_matrix_sensors.csv")
            self.CM_sensors = pd.read_csv("matrices/" + self.embedding + "_cm_sensors.csv")
            self.Eucli_Matrix_sensors = pd.read_csv("matrices/" + self.embedding + "_euclidean_matrix_sensors.csv")

            self.Cossine_Matrix_sensors.index = [int(i) for i in self.Cossine_Matrix_sensors.index]
            self.CM_sensors.index = [int(i) for i in self.CM_sensors.index]
            self.CM_sensors.columns = [int(i) for i in self.CM_sensors.columns]
            self.Eucli_Matrix_sensors.index = [int(i) for i in self.Eucli_Matrix_sensors.index]

            #if(not(self.without_road_matrix)):
            columns_sensors = np.array(sorted(np.intersect1d(self.Eucli_Matrix_sensors.columns.values,np.intersect1d(self.Cossine_Matrix_sensors.columns.values, self.Road_Matrix_sensors.columns.values)),key=lambda x : int(x)))
            index_sensors = np.intersect1d(self.Eucli_Matrix_sensors.index.values,np.intersect1d(self.Cossine_Matrix_sensors.index.values, self.Road_Matrix_sensors.index.values))

            self.Road_Matrix_sensors = self.Road_Matrix_sensors.loc[index_sensors,columns_sensors]

            if(101 in self.sts_id):
                self.sts_id.remove(101)
            if(102 in self.sts_id):
                self.sts_id.remove(102)
            if(103 in self.sts_id):
                self.sts_id.remove(103)

            #self.LatLong["trajectory_id"] = self.LatLong.index

            self.Objects_loc = self.sts_id
            self.obj_loc = self.Objects_loc[0]

            self.validation_2 = False
              
    def sensor_matrix_creation(self):

        if(isinstance(self.Road_Matrix_sensors,str)):
            if("Road_Matrix_sensors.csv" in listdir("./matrices")):
                self.Road_Matrix_sensors = pd.read_csv("matrices/Road_Matrix_sensors.csv")
            else:

                self.road_matrix_creation()
                #with self.intrinsic_evaluation_output:

                #    self.intrinsic_evaluation_output.clear_output()

                #    road_label = widgets.Label("The Road Matrix is ​​not in a folder, would you like to generate it (it may take a long time)?")

                #    yes_button = widgets.Button(description="Yes", layout=widgets.Layout(width="100px"))
                #    yes_button.style.button_color = "lightgray"
                #    yes_button.on_click(self.road_matrix_yes)

                #    no_button = widgets.Button(description="No", layout=widgets.Layout(width="100px"))
                #    no_button.style.button_color = "lightgray"
                #    no_button.on_click(self.road_matrix_no)

                #    yes_no_box =  widgets.HBox([yes_button, self.spc1, no_button])

                #    road_box = widgets.VBox([road_label, self.spc1, yes_no_box])

                #    display(road_box)
    
        dist_matrix_coss_sensor = np.zeros((len(self.sts_id),len(self.sts_id)))
        eucl_distance = np.zeros((len(self.sts_id),len(self.sts_id)))


        for i in range(len(self.sts_id)):
            for j in range(len(self.sts_id)):
                dist_matrix_coss_sensor[i,j] = nltk.cluster.cosine_distance(np.array(self.embedding_matrix)[self.sts_id[i]-1],np.array(self.embedding_matrix)[self.sts_id[j]-1])


        lat_values = []


        for i in self.tokenizer_df.index:

            if(i not in [101,102,103] ):
                lat_values.append(self.LL.loc[self.tokenizer_df[self.tokenizer_df["id"] == i]["sensor"].apply(lambda x : x.lower())]["lat"].values[0])

            else:

                lat_values.append("n")

        lon_values = []

        for i in self.tokenizer_df.index:

            if(i not in [101,102,103]):

                lon_values.append(self.LL.loc[self.tokenizer_df[self.tokenizer_df["id"] == i]["sensor"].apply(lambda x : x.lower())]["lon"].values[0])

            else:

                lon_values.append("n")    

    
        for i in range(len(lat_values)):
            for j in range(len(lat_values)):
                if((lat_values[i] != "n") and (lat_values[j] != "n") ):     
                    eucl_distance[i,j] = distances.euclidean_distance_in_meters(lat_values[i],lon_values[i],lat_values[j],lon_values[j])



        self.Cossine_Matrix_sensors = pd.DataFrame(dist_matrix_coss_sensor)

        self.Eucli_Matrix_sensors = pd.DataFrame(eucl_distance)

        for i in self.Cossine_Matrix_sensors.index:

            self.Cossine_Matrix_sensors.iloc[i,i] = np.nan 

        for i in self.Eucli_Matrix_sensors.index:

            self.Eucli_Matrix_sensors.iloc[i,i] = np.nan

        self.CM_sensors = copy(self.Cossine_Matrix_sensors)
        self.CM_sensors.columns = [int(i) for i in self.CM_sensors.columns]
        self.CM_sensors.to_csv("matrices/" + self.embedding + "_cm_sensors.csv",index=False)

        self.Cossine_Matrix_sensors.columns = [str(i) for i in self.sts_id]
        self.Eucli_Matrix_sensors.columns = [str(i) for i in self.sts_id]
        self.Cossine_Matrix_sensors.index = self.sts_id
        self.Eucli_Matrix_sensors.index = self.sts_id

        #if(not(self.without_road_matrix)):

        columns_sensors = np.array(sorted(np.intersect1d(self.Eucli_Matrix_sensors.columns.values,np.intersect1d(self.Cossine_Matrix_sensors.columns.values, self.Road_Matrix_sensors.columns.values)),key=lambda x : int(x)))
        index_sensors = np.intersect1d(self.Eucli_Matrix_sensors.index.values,np.intersect1d(self.Cossine_Matrix_sensors.index.values, self.Road_Matrix_sensors.index.values))

        self.Cossine_Matrix_sensors = self.Cossine_Matrix_sensors.loc[index_sensors,columns_sensors]

        self.Eucli_Matrix_sensors = self.Eucli_Matrix_sensors.loc[index_sensors,columns_sensors]

        self.Road_Matrix_sensors = self.Road_Matrix_sensors.loc[index_sensors,columns_sensors]

        self.Cossine_Matrix_sensors.to_csv("matrices/" + self.embedding + "_cossine_matrix_sensors.csv",index=False)
        self.Eucli_Matrix_sensors.to_csv("matrices/" + self.embedding + "_euclidean_matrix_sensors.csv",index=False)



        #self.LatLong["trajectory_id"] = self.LatLong.index


        if(101 in self.sts_id):
            self.sts_id.remove(101)
        if(102 in self.sts_id):
            self.sts_id.remove(102)
        if(103 in self.sts_id):
            self.sts_id.remove(103)

        self.Objects_loc = self.sts_id
        self.obj_loc = self.Objects_loc[0]
          
     
    def on_change_traj_data(self, change):

            # When the sequential data set is selected, the screen is cleared and then the calculations are made, modifying the class attributes, only then is the screen recomposed
            with self.text_dataset_out:
                
                self.text_dataset_out.clear_output()
                
                display(widgets.Label("Processing..."))
            
            
            self.traj_data = change.new
            
            if(self.traj_data != ''):      
                
                self.data_traj_list_dict()
                
                self.traj_matrix_read()
                
                if(self.validation):
            
                    self.traj_matrix_creation()

                else:

                    self.validation = True
                
                self.sensor_matrix_read()
                
                if(self.validation_2):
                    
                    self.sensor_matrix_creation()
                
                else:

                    self.validation_2 = True
                
                # The dropdown locations for plotting the top-k locations are only modified when a new sequential dataset is selected.
                with self.select_location_dropdown_output:
                    
                    self.select_location_dropdown_output.clear_output()
                    self.select_location_dropdown.options = self.Objects_loc
                    self.select_location_dropdown.value = self.obj_loc
                    display(self.select_location_dropdown)
                
            else:
                
                pass
                
        
            with self.text_dataset_out:
            
                self.text_dataset.options = [''] if(not(self.embedding in self.Emb_dict.keys())) else self.Emb_dict[self.embedding] + ['']
                self.text_dataset.value = ''
            
                self.text_dataset_out.clear_output()
                
                display(self.text_dataset)
            

    def traj_plot_mrr(self, mrr_trajectory_plot_button):
            
        
            with self.out_mrr_traj:
                
                self.out_mrr_traj.layout = widgets.Layout(width = "450px",border='solid 2.0px white',margin='0px 5px 5px 5px',padding='2px 2px 2px 2px')
                self.out_mrr_traj.clear_output()
                display(widgets.Label("Processing..."))


                mrrdtw=[]
                for dtw in range(5,100,5): 
                    mrr_ = mean_reciprocal_rank_filter(self.Cossine_Matrix.loc[self.DTW_Matrix.index.values,:], self.DTW_Matrix , dtw, dtw+1,2)
                    mrrdtw.append(mrr_)
                    
                

                mrredit=[]
                for d in range(5,100,5): 
                    mrr_ = mean_reciprocal_rank_filter(self.Cossine_Matrix, self.Edit_Matrix, d, d+1,2)
                    mrredit.append(mrr_)
                    
                

                plt.plot( range(5,100,5), mrrdtw, '-*')
                plt.plot( range(5,100,5), mrredit, '-o')

                plt.legend(['DTW', 'Edit distance'])
                plt.xlabel('distance')
                plt.ylabel('mrr')
                plt.title('Mean Reciprocal Ranking vs Maximal Distance')
                
                self.out_mrr_traj.clear_output()
                
                plt.show()
                
                self.out_mrr_traj.layout = widgets.Layout(width = "450px",border='solid 2.0px white',margin='0px 5px 5px 5px',padding='2px 2px 2px 2px')
                
    def loc_mrr_plot(self, mrr_location_plot_button ):
            
            with self.out_mrr_loc:
                
                self.out_mrr_loc.layout = widgets.Layout(width = "450px",border='solid 2.0px white',margin='0px 5px 5px 5px',padding='2px 2px 2px 2px')
                self.out_mrr_loc.clear_output()
                display(widgets.Label("Processing..."))
                
                mrre=[]
                for ed in range(500,10001,500):
                    mrr_ = mean_reciprocal_rank_filter(self.Cossine_Matrix_sensors, self.Eucli_Matrix_sensors, ed,ed+1,2)
                    mrre.append(mrr_)


                mrr=[]
                for rd in range(500,10001,500):
                    mrr_ = mean_reciprocal_rank_filter(self.Cossine_Matrix_sensors, self.Road_Matrix_sensors, rd,rd+1,2)
                    mrr.append(mrr_)

                plt.plot( range(500,10001,500), mrr, '-*')

                plt.plot( range(500,10001,500), mrre, '-o')

                plt.legend(['Road Distance', 'Euclidean Euclidean'])
                plt.xlabel('distance (meters)')
                plt.ylabel('mrr')
                plt.title('Mean Reciprocal Ranking vs Range Distance')
                plt.savefig('mrr_correct',format='pdf')
                
                
                self.out_mrr_loc.clear_output()
                
                plt.show()
                
                self.out_mrr_loc.layout = widgets.Layout(width = "450px",border='solid 2.0px white',margin='0px 5px 5px 5px',padding='2px 2px 2px 2px')
                          
    def traj_topk_plot(self, top_k_trajectory_button):
            
            
            with self.out_topk_traj:
                
                self.out_topk_traj.clear_output()
                
                display(widgets.Label("Processing..."))
            
                traj_matrix_index = self.Dataset[self.Dataset["trajectory_number"] == str(self.obj_traj)].index[0]
                k = int(self.topk_traj)
                
                if(isinstance(self.CM, pd.DataFrame)):
                    self.CM = self.CM.to_numpy() 
                
                neighborhood_topk = []
                for i in range(self.CM.shape[0]):
                    neighborhood_topk.append((self.CM[i][traj_matrix_index], i))

                neighborhood_topk_index = sorted(neighborhood_topk)

                neighborhood_topk_index = neighborhood_topk_index[0:k+1]
                
                
                trajectories = [self.Dataset["trajectory_number"].loc[nb[1]] for nb in neighborhood_topk_index]
                

                trajectories_id = list(self.Traj_Number_Traj_id["0"].loc[[tj for tj in trajectories]])

                
                trajectories_df = pd.concat([filters.by_label(self.LatLong, value = tj, label_name = "trajectory_id") for tj in trajectories_id],axis=0)

                trajectories_df['trajectory_id'] = trajectories_df['trajectory_id'].apply(lambda x : "Trajectory : "+str(self.traj_id_dict[x]))

                move_df = MoveDataFrame(data= trajectories_df, latitude="lat", longitude="lon", datetime="time", traj_id='trajectory_id')
                
                self.out_topk_traj.clear_output()
                
                display(f.plot_trajectories(move_df))
                    
    def loc_topk_plot(self, top_k_location_button):

        with self.out_topk_loc:

            self.out_topk_loc.clear_output()

            neighborhood_topk_s = []

            for index in self.CM_sensors.index:
                self.CM_sensors.loc[index][index] = 0.0

            cossine_matrix_s = self.CM_sensors.to_numpy() 

            for i in range(cossine_matrix_s.shape[0]):
                neighborhood_topk_s.append((cossine_matrix_s[i][self.sts_id.index(int(self.obj_loc))], i)) 

            neighborhood_topk_s_index = sorted(neighborhood_topk_s)
            neighborhood_topk_s_index = neighborhood_topk_s_index[0:int(self.topk_loc)+1]

            closest_labels = self.tokenizer_df.loc[[n[1] + 1 for n in neighborhood_topk_s_index], :]
            aux_cl = closest_labels['sensor'].apply(lambda x : x.lower())

            closest_sensor = self.LL.loc[aux_cl]
            closest_sensor['id'] = ["Location: "+str(n[1] + 1) for n in neighborhood_topk_s_index]
            closest_sensor['datetime'] = 0.0


            map = plot_trajectories(closest_sensor)

            self.out_topk_loc.clear_output()

            display(plot_points(closest_sensor, user_point='gray', base_map=map))