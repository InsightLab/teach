import ipywidgets as widgets
from IPython.display import display, HTML

import pandas as pd

import osmnx as ox
import networkx as nx

from time import sleep
from tqdm import trange
from random import sample 
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

        
def on_change_object_traj(change, object_traj):

    object_traj = change.new

def on_change_object_loc(change, object_loc):

    object_loc = change.new

def on_change_topk_loc(change, topk_loc):

    topk_loc = change.new

def on_change_topk_traj(change, topk_traj):

    topk_traj = change.new
    
def on_change_embedding_choice(change, embedding, embeddding_matrix, emb_dict, tokenizer_df, text_dataset, out):
        
        embedding = change.new
            
        aux_emb = pd.read_csv("embeddings/" + embedding + ".csv") 
    
        tokenizer_df = aux_emb.loc[0:list(pd.isnull(aux_emb["sensor"])).index(True)-1,["sensor","id"]]
        tokenizer_df.id = [int(i) for i in tokenizer_df.id]
        tokenizer_df.index = tokenizer_df.id
        tokenizer_df = tokenizer_df[["sensor","id"]]

        embeddding_matrix = aux_emb.loc[list(pd.isnull(aux_emb["sensor"])).index(True):int(list(pd.isnull(aux_emb["sensor"])).index(True)*2),
                        [str(i) for i in range(len(aux_emb.columns)-2)]]
        
        embeddding_matrix.index = [i for i in range(embeddding_matrix.shape[0])]
        
        with out:
            
            out.clear_output()
            
            display(widgets.Label("Processing..."))
        
            text_dataset.options = [''] if(not(embedding in emb_dict.keys())) else emb_dict[embedding] + ['']
            
            text_dataset.value = ''
            
            out.clear_output()
            
            display(text_dataset)


def embedding_not_linked(embedding_list):

    aux_emb = pd.read_csv("embeddings/" + embedding_list[0] + ".csv") 

    tokenizer_df = aux_emb.loc[0:list(pd.isnull(aux_emb["sensor"])).index(True)-1,["sensor","id"]]
    tokenizer_df.id = [int(i) for i in tokenizer_df.id]
    tokenizer_df.index = tokenizer_df.id
    tokenizer_df = tokenizer_df[["sensor","id"]]

    embedding_matrix = aux_emb.loc[list(pd.isnull(aux_emb["sensor"])).index(True):int(list(pd.isnull(aux_emb["sensor"])).index(True)*2),
                    [str(i) for i in range(len(aux_emb.columns)-2)]]

    embedding_matrix.index = [i for i in range(embedding_matrix.shape[0])]
    
    LL = pd.read_csv("lat_lon_sensors/"+"sensors_roubados_representativos_countmin18(1).csv", index_col='geos25')
    LL = LL.astype({'lat':'float32', 'lon':'float32'})
    LL.drop_duplicates(inplace=True)
    LL['sensor'] = [str(geos25).lower() for geos25 in LL.index.values]
    LL.set_index('sensor', inplace=True)

    return LL , tokenizer_df, embedding_matrix

def embedding_linked(embedding_list, embedding_dict):

    aux_emb = pd.read_csv("embeddings/" + embedding_list[0] + ".csv") 

    tokenizer_df = aux_emb.loc[0:list(pd.isnull(aux_emb["sensor"])).index(True)-1,["sensor","id"]]
    tokenizer_df.id = [int(i) for i in tokenizer_df.id]
    tokenizer_df.index = tokenizer_df.id
    tokenizer_df = tokenizer_df[["sensor","id"]]

    embedding_matrix = aux_emb.loc[list(pd.isnull(aux_emb["sensor"])).index(True):int(list(pd.isnull(aux_emb["sensor"])).index(True)*2), [str(i) for i in range(len(aux_emb.columns)-2)]]

    embedding_matrix.index = [i for i in range(embedding_matrix.shape[0])]
    
    traj_data = embedding_dict[embedding_list[0]][0]
    
    embedding = embedding_list[0]
    
    LatLong   = pd.read_csv("data/"+traj_data+".csv") 
    aux_latlon = sorted(list(set(LatLong["trajectory_id"])))
    
    LL = pd.read_csv("lat_lon_sensors/"+"sensors_roubados_representativos_countmin18(1).csv", index_col='geos25')
    LL = LL.astype({'lat':'float32', 'lon':'float32'})
    LL.drop_duplicates(inplace=True)
    LL['sensor'] = [str(geos25).lower() for geos25 in LL.index.values]
    LL.set_index('sensor', inplace=True)
    
    t_id_list = []
    Sensors = list(tokenizer_df["sensor"].apply(lambda x : x.upper()))
    
    
    for tid in aux_latlon:
        
        traj = list(LatLong[LatLong["trajectory_id"] == tid]["location_label"])
        
        in_dict = True
        
        for sensor in traj:
            
            if(not(sensor in Sensors)):
                in_dict = False
        
        if(in_dict):
            
            t_id_list.append(tid)
            
    
    t_id_list = sorted(t_id_list)
    traj_id_dict_all = {t_id_list[i]: i for i in range(len(t_id_list))}

    return LL, LatLong, tokenizer_df, embedding_matrix, embedding, embedding_dict, embedding_list, t_id_list, traj_id_dict_all, traj_data

def data_sample_traj_matrix_first_read(traj_data, embedding, traj_id_dict_all, tokenizer_df, embedding_matrix, query_index):
    
    for file in listdir("./data_samples"):
            
        if((traj_data in file) and (embedding in file)):

            traj_data_sample = file.split(".")[0]
            lat_long_2_sample = pd.read_csv("data_samples/" + file) 
            t_id_list_sample = sorted(list(set(lat_long_2_sample["trajectory_id"])))
            dict_traj_id = {tid: traj_id_dict_all[tid] for tid in t_id_list_sample}
            objects_traj = sorted(list(dict_traj_id.values()))
            obj_traj = objects_traj[0]

            trajs_token = sorted(dict_traj_id.keys())

            dataset_traj_number = pd.read_csv("trajectories/" + traj_data_sample + "_trajs"+".csv") 
            
            dataset_traj_number["0"] = dataset_traj_number["0"].apply(lambda x : x.replace("[","").replace("]","").replace(",","").replace("\n","").replace("'",""))
            dataset_traj_number["0"] = dataset_traj_number["0"].apply(lambda x : x.split())
            dataset_traj_number["trajectory_number"] = dataset_traj_number["trajectory_number"].apply(lambda x: str(x))

            traj_number = pd.DataFrame({"0":trajs_token},index=[str(dict_traj_id[i]) for i in trajs_token])

            dataset_traj_number_geo25 = copy(dataset_traj_number)

            dict_geo25 = { geo25.upper():vector  for geo25,vector in zip(list(tokenizer_df['sensor'].values),np.array(embedding_matrix))}

            dataset_traj_number["0"] = dataset_traj_number["0"].apply(lambda x:[ dict_geo25[geo25] for geo25 in x ]) 
            dataset_traj_number["0"] = dataset_traj_number["0"].apply(lambda x : np.array(sum(x)/len(x)))

            
            cossine_matrix_1 = pd.read_csv("matrices/"  + traj_data_sample+"_cossine_matrix_trajs.csv")   
            cossine_matrix_2 = pd.read_csv("matrices/"  + traj_data_sample+"_cm_trajs.csv")
            dtw_matrix = pd.read_csv("matrices/"  + traj_data_sample+"_dtw_matrix_trajs.csv")           
            edit_matrix = pd.read_csv("matrices/" +  traj_data_sample+"_edit_matrix_trajs.csv") 

            query_index = np.arange(0,cossine_matix_1.shape[0])

            for index in dataset_traj_number_geo25["0"].index:
                for s in dataset_traj_number_geo25["0"].loc[index]:
                    sensors_traj_sample.append(s.lower())
                
            sensors_traj_sample = list(set(sensors_traj_sample))

            sample_sensor_id_list = []
            
            for index in tokenizer_df.index:
                if(tokenizer_df.sensor.loc[index].lower() in sensors_traj_sample):
                    sample_sensor_id_list.append(index)

                elif(tokenizer_df.sensor.loc[index]=="[CSL]" or tokenizer_df.sensor.loc[index]=="[SEP]" or tokenizer_df.sensor.loc[index]=="[MASK]"):
                    sample_sensor_id_list.append(index)

            return False, query_index, traj_data_sample, lat_long_2_sample, dict_traj_id, objects_traj, obj_traj, dataset_traj_number, dataset_traj_number_geo25, dict_geo25, traj_number, sample_sensor_id_list, cossine_matrix_1, cossine_matrix_2, dtw_matrix, edit_matrix
            
    return True, query_index, traj_data_sample, lat_long_2_sample, dict_traj_id, objects_traj, obj_traj, dataset_traj_number, dataset_traj_number_geo25, dict_geo25, traj_number, sample_sensor_id_list, cossine_matrix_1, cossine_matrix_2, dtw_matrix, edit_matrix

def data_sample_traj_matrix_first_creation(embedding, traj_data, t_id_list, LatLong, traj_id_dict_all, tokenizer_df, embedding_matrix):
    
    traj_data_sample = embedding + "_" + traj_data + "_sample"   
    t_id_list_sample = sorted(sample(t_id_list,50)) 
    
    
    time = []
    lat = []
    lon = []
    location_label = []
    trajectory_id = []

    for i in range(LatLong.shape[0]):

        if(LatLong["trajectory_id"].loc[i] in t_id_list_sample):

            time.append(LatLong["time"].loc[i])
            lat.append(LatLong["lat"].loc[i])
            lon.append(LatLong["lon"].loc[i])
            location_label.append(LatLong["location_label"].loc[i])
            trajectory_id.append(LatLong["trajectory_id"].loc[i])


    lat_long_sample_dict = {"trajectory_id":trajectory_id,"time":time, "lat":lat, "lon":lon,"location_label":location_label}
    lat_long_sample = pd.DataFrame(lat_long_sample_dict)

    
    lat_long_sample.to_csv("data_samples/"+traj_data_sample+".csv",index=False)
    dict_traj_id = {tid: traj_id_dict_all[tid] for tid in t_id_list_sample}
    objects_traj = sorted(list(dict_traj_id.values()))
    object_traj = objects_traj[0]

    trajs_token = sorted(dict_traj_id.keys())
    traj_sample_dict = {"0":[list(lat_long_sample[lat_long_sample["trajectory_id"] == traj]["location_label"]) for traj in trajs_token],"trajectory_number":[str(dict_traj_id[i]) for i in trajs_token]}
    dataset_traj_number = pd.DataFrame(traj_sample_dict,index=[str(dict_traj_id[i]) for i in trajs_token])
    dataset_traj_number.to_csv("data/"+traj_data_sample + "_trajs"+".csv",index=False)

    traj_number = pd.DataFrame({"0":trajs_token},index=[str(dict_traj_id[i]) for i in trajs_token])

    dataset_traj_number_geo25  = copy(dataset_traj_number)

    dict_geo25 = { geo25.upper():vector  for geo25, vector in zip(list(tokenizer_df['sensor'].values),np.array(embedding_matrix))}
    
    
    dataset_traj_number["0"] = dataset_traj_number["0"].apply(lambda x:[ dict_geo25[geo25] for geo25 in x ]) 
    dataset_traj_number["0"] = dataset_traj_number["0"].apply(lambda x : np.array(sum(x)/len(x)))

    dataset_traj_number.index = [i for i in range(len(dataset_traj_number))]

    dist_matrix_coss_traj = np.zeros((dataset_traj_number["0"].shape[0],dataset_traj_number["0"].shape[0]))

    for i in range(dataset_traj_number["0"].shape[0]):
        for j in range(dataset_traj_number["0"].shape[0]):
            dist_matrix_coss_traj[i][j] = nltk.cluster.cosine_distance(dataset_traj_number["0"][i],dataset_traj_number["0"][j])


    cossine_matix_1 = pd.DataFrame(dist_matrix_coss_traj)

    query_index = np.arange(0,cossine_matix_1.shape[0])


    cossine_matix_2 = copy(cossine_matix_1)
    cossine_matix_2.to_csv("matrices/" + traj_data_sample+"_cm_trajs.csv",index=False)
    
    for i in range(0,cossine_matix_1.shape[0]):

        cossine_matix_1.iloc[i,i] = np.nan 
        
    cossine_matix_1.to_csv("matrices/"  + traj_data_sample + "_cossine_matrix_trajs.csv",index=False)
    
    
    matrix_dist_dtw = np.zeros((len(t_id_list_sample),len(t_id_list_sample)), dtype='float16')

    TJ_ID = t_id_list_sample

    for i in range(len(TJ_ID)):
        for j in range(i,len(TJ_ID)):
            matrix_dist_dtw[i][j] = dtw_distance(TJ_ID[i], TJ_ID[j], lat_long_sample)
            
    for i in reversed(range(len(TJ_ID))):
        for j in reversed(range(i)):
            matrix_dist_dtw[i][j] = matrix_dist_dtw[j][i]
            
    dtw_matrix = pd.DataFrame(matrix_dist_dtw)


    dist_matrix = -np.ones((dataset_traj_number_geo25 .shape[0],dataset_traj_number_geo25 .shape[0]), dtype='float16')

    ids_list = list(dataset_traj_number_geo25 ["0"].index)
    for i in range(len(ids_list)):
        for j in range(len(ids_list)):
            dist_matrix[i][j] = edit_distance(dataset_traj_number_geo25 ["0"].loc[ids_list[i]],dataset_traj_number_geo25 ["0"].loc[ids_list[j]])
            
    edit_matrix = pd.DataFrame(dist_matrix)
    
    for i in range(0,dtw_matrix.shape[0]):

        dtw_matrix.iloc[i,i] = np.nan
        edit_matrix.iloc[i,i] = np.nan
        
        
    dtw_matrix.to_csv("matrices/"  + traj_data_sample +"_dtw_matrix_trajs.csv",index=False)
    edit_matrix.to_csv("matrices/" + traj_data_sample +"_edit_matrix_trajs.csv",index=False)
    
    for index in dataset_traj_number_geo25["0"].index:
        for s in dataset_traj_number_geo25 ["0"].loc[index]:
            sensors_traj_sample.append(s.lower())
        
    sensors_traj_sample = list(set(sensors_traj_sample))

    sample_sensor_id_list = []

    for index in tokenizer_df.index:
        if(tokenizer_df.sensor.loc[index].lower() in sensors_traj_sample):
            sample_sensor_id_list.append(index)

        elif(tokenizer_df.sensor.loc[index]=="[CSL]" or tokenizer_df.sensor.loc[index]=="[SEP]" or tokenizer_df.sensor.loc[index]=="[MASK]"):
            sample_sensor_id_list.append(index)
        
    return query_index, dict_traj_id, lat_long_sample, objects_traj , object_traj, dataset_traj_number, dataset_traj_number_geo25, dict_geo25, traj_number, cossine_matix_1, cossine_matix_2, dtw_matrix, edit_matrix, sample_sensor_id_list 

def sensor_matrix_first_read(embedding, road_matrix_sensors, sample_sensor_id_list, lat_long_2):

    for file in listdir("./matrices"):
        
        if((embedding in file) and not("sample" in file)):
            
            cossine_matrix_sensors_1 = pd.read_csv("matrices/" + embedding + "_cossine_matrix_sensors.csv")
            cossine_matrix_sensors_2 = pd.read_csv("matrices/" + embedding + "_cm_sensors.csv")
            euclidean_matrix_sensors = pd.read_csv("matrices/" + embedding + "_euclidean_matrix_sensors.csv")
            
            cossine_matrix_sensors_1.index = [int(i) for i in cossine_matrix_sensors_1.columns]
            cossine_matrix_sensors_2.index = [int(i) for i in cossine_matrix_sensors_2.columns]
            cossine_matrix_sensors_2.columns = [int(i) for i in cossine_matrix_sensors_2.columns]
            euclidean_matrix_sensors.index = [int(i) for i in euclidean_matrix_sensors.columns]
            
            columns_sensors = np.array(sorted(np.intersect1d(euclidean_matrix_sensors.columns.values,np.intersect1d(cossine_matrix_sensors_1.columns.values, road_matrix_sensors.columns.values)),key=lambda x : int(x)))
            index_sensors = np.intersect1d(euclidean_matrix_sensors.index.values,np.intersect1d(cossine_matrix_sensors_1.index.values, road_matrix_sensors.index.values))
            road_matrix_sensors = road_matrix_sensors.loc[index_sensors,columns_sensors]
            
            lat_long_2["trajectory_id"] = lat_long_2.index
            
            if(101 in sample_sensor_id_list):
                sample_sensor_id_list.remove(101)
            if(102 in sample_sensor_id_list):
                sample_sensor_id_list.remove(102)
            if(103 in sample_sensor_id_list):
                sample_sensor_id_list.remove(103)

            Objects_loc = sample_sensor_id_list
            obj_loc = Objects_loc[0]
            
            return False, cossine_matrix_sensors_1, cossine_matrix_sensors_2, euclidean_matrix_sensors, road_matrix_sensors, Objects_loc, obj_loc, sample_sensor_id_list, lat_long_2

    return True, cossine_matrix_sensors_1, cossine_matrix_sensors_2, euclidean_matrix_sensors, road_matrix_sensors, Objects_loc, obj_loc, sample_sensor_id_list, lat_long_2      

def sensor_matrix_first_creation(embedding, road_matrix_sensors, tokenizer_df, sample_sensor_id_list, embedding_matrix, lat_long_2, objects_loc, obj_loc):

    dist_matrix_coss_sensor = np.zeros((len(sample_sensor_id_list),len(sample_sensor_id_list)))
    eucl_distance = np.zeros((len(sample_sensor_id_list),len(sample_sensor_id_list)))

    for i in range(len(sample_sensor_id_list)):
        for j in range(len(sample_sensor_id_list)):
            dist_matrix_coss_sensor[i,j] = nltk.cluster.cosine_distance(np.array(embedding_matrix)[sample_sensor_id_list[i]-1],np.array(embedding_matrix)[sample_sensor_id_list[j]-1])


    tokenizer_df_sample = tokenizer_df.loc[sample_sensor_id_list]


    lat_values = []


    for i in tokenizer_df_sample.index:

        if(i not in [101,102,103] ):
            lat_values.append(LL.loc[tokenizer_df_sample[tokenizer_df_sample["id"] == i]["sensor"].apply(lambda x : x.lower())]["lat"].values[0])

        else:
            lat_values.append("n")

    lon_values = []

    for i in tokenizer_df_sample.index:

        if(i not in [101,102,103]):
            lon_values.append(LL.loc[tokenizer_df_sample[tokenizer_df_sample["id"] == i]["sensor"].apply(lambda x : x.lower())]["lon"].values[0])

        else:
            lon_values.append("n")    


    for i in range(len(lat_values)):
        for j in range(len(lat_values)):
            if((lat_values[i] != "n") and (lat_values[j] != "n") ):     
                eucl_distance[i,j] = distances.euclidean_distance_in_meters(lat_values[i],lon_values[i],lat_values[j],lon_values[j])



    cossine_matrix_sensors_1 = pd.DataFrame(dist_matrix_coss_sensor)

    cossine_matrix_sensors_2 = copy(cossine_matrix_sensors_1)
    cossine_matrix_sensors_2.to_csv("matrices/" + embedding + "_cm_sensors.csv",index=False)

    euclidean_matrix_sensors = pd.DataFrame(eucl_distance)

    for i in cossine_matrix_sensors_1.index:

        cossine_matrix_sensors_1.iloc[i,i] = np.nan 

    for i in euclidean_matrix_sensors.index:

        euclidean_matrix_sensors.iloc[i,i] = np.nan

    cossine_matrix_sensors_1.columns = [str(i) for i in sample_sensor_id_list]
    euclidean_matrix_sensors.columns = [str(i) for i in sample_sensor_id_list]
    cossine_matrix_sensors_1.index = sample_sensor_id_list
    euclidean_matrix_sensors.index = sample_sensor_id_list

    columns_sensors = np.array(sorted(np.intersect1d(euclidean_matrix_sensors.columns.values,np.intersect1d(cossine_matrix_sensors_1.columns.values, road_matrix_sensors.columns.values)),key=lambda x : int(x)))
    index_sensors = np.intersect1d(euclidean_matrix_sensors.index.values,np.intersect1d(cossine_matrix_sensors_1.index.values, road_matrix_sensors.index.values))
    
    cossine_matrix_sensors_1 = cossine_matrix_sensors_1.loc[index_sensors,columns_sensors]
    euclidean_matrix_sensors = euclidean_matrix_sensors.loc[index_sensors,columns_sensors]
    road_matrix_sensors = road_matrix_sensors.loc[index_sensors,columns_sensors]

    cossine_matrix_sensors_1.to_csv("matrices/" + embedding + "_cossine_matrix_sensors.csv",index=False)
    euclidean_matrix_sensors.to_csv("matrices/" + embedding + "_euclidean_matrix_sensors.csv",index=False)
    
    cossine_matrix_sensors_2 = copy(cossine_matrix_sensors_1)
    cossine_matrix_sensors_2.columns = [int(i) for i in cossine_matrix_sensors_2.columns]
    cossine_matrix_sensors_2.to_csv("matrices/" + embedding + "_cm_sensors.csv",index=False)
      

    lat_long_2["trajectory_id"] =  lat_long_2.index
    
    if(101 in sample_sensor_id_list):
        sample_sensor_id_list.remove(101)
    if(102 in sample_sensor_id_list):
        sample_sensor_id_list.remove(102)
    if(103 in sample_sensor_id_list):
        sample_sensor_id_list.remove(103)
    
    objects_loc = sample_sensor_id_list 
    obj_loc = objects_loc[0]

    return cossine_matrix_sensors_1, cossine_matrix_sensors_2, euclidean_matrix_sensors, road_matrix_sensors, lat_long_2,  objects_loc, obj_loc, sample_sensor_id_list

def road_matrix_creation(out, tokenizer_df, sample_sensor_id_list, lat_long):

    out.clear_output()
    display(widgets.Label("Generating Road Matrix ..."))
    
    id_road = [] 
    sample_sensors = list(tokenizer_df.loc[sample_sensor_id_list]["sensor"].apply(lambda x : x.upper()).values)

    for index in lat_long.index:
        if(index.upper() in sample_sensors):
            id_road.append(index)

    lat_long_sample = lat_long.loc[id_road]

    south, west = lat_long_sample[['lat','lon']].min()
    north, east = lat_long_sample[['lat','lon']].max()

    g = ox.graph_from_bbox(north, south, east, west, network_type='drive_service')

    id_remove = []
    for index in lat_long_sample.index:
        if(not(lat_long_sample.loc[index].nodeId in g.nodes)):      
            id_remove.append(index)

    lat_long_sample = lat_long_sample.drop(id_remove)

    id_w_duplicate = []
    node_id_w_duplicate = []

    for index in lat_long_sample.index:
        if(not(lat_long_sample.loc[index].nodeId in node_id_w_duplicate)):      
            id_w_duplicate.append(index)
            node_id_w_duplicate.append(lat_long_sample.loc[index].nodeId)

    lat_long_sample = lat_long_sample.loc[id_w_duplicate]

    lat_long_sample = lat_long_sample.sort_values(by=['token_id'])

    net_dist = np.zeros((len(lat_long_sample.index.values),len(lat_long_sample.index.values)))
    node_list = [lat_long_sample.loc[lat_long_sample.index[s]]['nodeId'] for s in range(len(lat_long_sample.index))]

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
    df_dist.index = list(lat_long_sample.token_id)
    df_dist.columns = list(lat_long_sample.token_id)

    Road_Matrix_sensors = df_dist
    Road_Matrix_sensors.columns = [str(i) for i in Road_Matrix_sensors.columns]
    Road_Matrix_sensors.to_csv("matrices/" + "Road_Matrix_sensors.csv",index=False)
    
    out.clear_output()
    display(widgets.Label("Processing..."))

    return Road_Matrix_sensors

def data_traj_list_dict(lat_long_2, traj_data, tokenizer_df):
    
    lat_long_2   = pd.read_csv("data/"+traj_data+".csv") 
    aux = sorted(list(set(lat_long_2["trajectory_id"])))

    t_id_list = []
    Sensors = list(tokenizer_df["sensor"].apply(lambda x : x.upper()))


    for tid in list(set(lat_long_2["trajectory_id"])):

        traj = list(lat_long_2[lat_long_2["trajectory_id"] == tid]["location_label"])

        in_dict = True

        for sensor in traj:

            if(not(sensor in Sensors)):
                in_dict = False

        if(in_dict):

            t_id_list.append(tid)

    
    return sorted(t_id_list), {t_id_list[i]: i for i in range(len(t_id_list))}, lat_long_2



def data_sample_traj_matrix_read(validation, embedding, embedding_matrix, traj_data, traj_data_sample, lat_long_2_sample, dict_traj_id, objects_traj, object_traj,dataset_traj_number, dataset_traj_number_geo25, 
                     cossine_matrix_1, cossine_matrix_2, dtw_matrix, edit_matrix,  traj_id_dict_all, sample_sensor_id_list, sensors_traj_sample, tokenizer_df, select_trajectory_dropdown, out2):                                                                                                                                                                                                                                                                                       
    
    for file in listdir("./data_samples"):
        
        if(traj_data in file and embedding in file):

            traj_data_sample = file.split(".")[0]
            lat_long_2_sample = pd.read_csv("data_samples/" + file) 
            t_id_list_sample = sorted(list(set(lat_long_2_sample["trajectory_id"])))
            dict_traj_id = {tid: traj_id_dict_all[tid] for tid in t_id_list_sample}
            objects_traj = sorted(list(dict_traj_id.values()))
            object_traj = objects_traj[0]

            with out2:

                select_trajectory_dropdown.options = objects_traj
                select_trajectory_dropdown.value = object_traj

            trajs_token = sorted(dict_traj_id.keys())

            dataset_traj_number = pd.read_csv("trajectories/" + traj_data_sample + "_trajs"+".csv") 

            dataset_traj_number["0"] = dataset_traj_number["0"].apply(lambda x : x.replace("[","").replace("]","").replace(",","").replace("\n","").replace("'",""))
            dataset_traj_number["0"] = dataset_traj_number["0"].apply(lambda x : x.split())
            dataset_traj_number["trajectory_number"] = dataset_traj_number["trajectory_number"].apply(lambda x: str(x))

            traj_number = pd.DataFrame({"0":trajs_token},index=[str(dict_traj_id[i]) for i in trajs_token])

            dataset_traj_number_geo25 = copy(dataset_traj_number)
            dict_geo25 = { geo25.upper():vector  for geo25,vector in zip(list(tokenizer_df['sensor'].values),np.array(embedding_matrix))}

            dataset_traj_number["0"] = dataset_traj_number["0"].apply(lambda x:[ dict_geo25[geo25] for geo25 in x ]) #No futuro verificar pra substituir
            dataset_traj_number["0"] = dataset_traj_number["0"].apply(lambda x : np.array(sum(x)/len(x)))


            cossine_matrix_1 = pd.read_csv("matrices/" +  traj_data_sample + "_cossine_matrix_trajs.csv")
            cossine_matrix_2 = pd.read_csv("matrices/" + traj_data_sample + "_cm_trajs.csv")
            dtw_matrix = pd.read_csv("matrices/" + traj_data_sample + "_dtw_matrix_trajs.csv") 
            edit_matrix = pd.read_csv("matrices/" +  traj_data_sample + "_edit_matrix_trajs.csv") 

            for index in dataset_traj_number_geo25["0"].index:
                for s in dataset_traj_number_geo25["0"].loc[index]:
                    sensors_traj_sample.append(s.lower())

            sensors_traj_sample = list(set(sensors_traj_sample))
            sample_sensor_id_list  = []

            for index in tokenizer_df.index:
                if(tokenizer_df.sensor.loc[index].lower() in sensors_traj_sample):
                    sample_sensor_id_list.append(index)

                elif(tokenizer_df.sensor.loc[index]=="[CSL]" or tokenizer_df.sensor.loc[index]=="[SEP]" or tokenizer_df.sensor.loc[index]=="[MASK]"):
                    sample_sensor_id_list.append(index)

            return False, traj_number, traj_data_sample, lat_long_2_sample, dict_traj_id, objects_traj, object_traj, dataset_traj_number, dataset_traj_number_geo25, cossine_matrix_1, cossine_matrix_2, dtw_matrix, edit_matrix, sample_sensor_id_list, sensors_traj_sample, out2
        
    return True, traj_data_sample, lat_long_2_sample, dict_traj_id, objects_traj, object_traj, dataset_traj_number, dataset_traj_number_geo25, cossine_matrix_1, cossine_matrix_2, dtw_matrix, edit_matrix, sample_sensor_id_list, sensors_traj_sample, out2
            
def data_sample_traj_matrix_creation(sensors_traj_sample, embedding, embedding_matrix, edit_matrix, dtw_matrix, cossine_matrix_1, dataset_traj_number_geo25, traj_number, dataset_traj_number, object_traj, objects_traj, dict_traj_id, lat_long_2_sample, 
                                            lat_long_2, traj_data, t_id_list, traj_id_dict_all, traj_data_sample, tokenizer_df, select_trajectory_dropdown, out2):
    
    traj_data_sample = embedding + "_" + traj_data + "_sample"   
    t_id_list_sample = sorted(sample(t_id_list,50)) 


    time = []
    lat = []
    lon = []
    location_label = []
    trajectory_id = []

    for i in range(lat_long_2.shape[0]):

        if(lat_long_2["trajectory_id"].loc[i] in t_id_list_sample):

            time.append(lat_long_2["time"].loc[i])
            lat.append(lat_long_2["lat"].loc[i])
            lon.append(lat_long_2["lon"].loc[i])
            location_label.append(lat_long_2["location_label"].loc[i])
            trajectory_id.append(lat_long_2["trajectory_id"].loc[i])


    lat_long_2_sample_dict = {"trajectory_id":trajectory_id,"time":time, "lat":lat, "lon":lon,"location_label":location_label}
    lat_long_2_sample = pd.DataFrame(lat_long_2_sample_dict)


    lat_long_2_sample.to_csv("data_samples/"+traj_data_sample+".csv",index=False)
    dict_traj_id = {tid: traj_id_dict_all[tid] for tid in t_id_list_sample}
    objects_traj = sorted(list(dict_traj_id.values()))
    object_traj = objects_traj[0]

    with out2:

            select_trajectory_dropdown.options = objects_traj
            

    trajs_token = sorted(dict_traj_id.keys())
    traj_sample_dict = {"0":[list(lat_long_2_sample[lat_long_2_sample["trajectory_id"] == traj]["location_label"]) for traj in trajs_token], "trajectory_number":[str(dict_traj_id[i]) for i in trajs_token]}
    dataset_traj_number = pd.DataFrame(traj_sample_dict,index=[str(dict_traj_id[i]) for i in trajs_token])
    dataset_traj_number.to_csv("trajectories/"+ traj_data_sample + "_trajs"+".csv",index=False)

    traj_number = pd.DataFrame({"0":trajs_token},index=[str(dict_traj_id[i]) for i in trajs_token])

    dataset_traj_number_geo25 = copy(dataset_traj_number)

    dict_geo25 = { geo25.upper():vector  for geo25,vector in zip(list(tokenizer_df['sensor'].values),np.array(embedding_matrix))}


    dataset_traj_number["0"] = dataset_traj_number["0"].apply(lambda x:[ dict_geo25[geo25] for geo25 in x ]) 
    dataset_traj_number["0"] = dataset_traj_number["0"].apply(lambda x : np.array(sum(x)/len(x)))


    dataset_traj_number.index = [i for i in range(len(dataset_traj_number))]


    dist_matrix_coss_traj = np.zeros((dataset_traj_number["0"].shape[0],dataset_traj_number["0"].shape[0]))

    for i in range(dataset_traj_number["0"].shape[0]):
        for j in range(dataset_traj_number["0"].shape[0]):
            dist_matrix_coss_traj[i][j] = nltk.cluster.cosine_distance(dataset_traj_number["0"][i],dataset_traj_number["0"][j])


    cossine_matrix_1 = pd.DataFrame(dist_matrix_coss_traj)

    query_index = np.arange(0,cossine_matrix_1.shape[0])


    cossine_matrix_2 = copy(cossine_matrix_1)
    cossine_matrix_2.to_csv("matrices/" +  traj_data_sample + "_cm_trajs.csv",index=False)

    for i in range(0,cossine_matrix_1.shape[0]):

        cossine_matrix_1.iloc[i,i] = np.nan 

    cossine_matrix_1.to_csv("matrices/" +  traj_data_sample + "_cossine_matrix_trajs.csv",index=False)


    matrix_dist_dtw = np.zeros((len(t_id_list_sample),len(t_id_list_sample)), dtype='float16')

    TJ_ID = t_id_list_sample

    for i in range(len(TJ_ID)):
        for j in range(i,len(TJ_ID)):
            matrix_dist_dtw[i][j] = dtw_distance(TJ_ID[i], TJ_ID[j], lat_long_2_sample)

    for i in reversed(range(len(TJ_ID))):
        for j in reversed(range(i)):
            matrix_dist_dtw[i][j] = matrix_dist_dtw[j][i]

    dtw_matrix = pd.DataFrame(matrix_dist_dtw)


    dist_matrix = -np.ones((dataset_traj_number_geo25.shape[0],dataset_traj_number_geo25.shape[0]), dtype='float16')

    ids_list = list(dataset_traj_number_geo25["0"].index)
    for i in range(len(ids_list)):
        for j in range(len(ids_list)):
            dist_matrix[i][j] = edit_distance(dataset_traj_number_geo25["0"].loc[ids_list[i]],dataset_traj_number_geo25["0"].loc[ids_list[j]])

    edit_matrix = pd.DataFrame(dist_matrix)


    for i in range(0,dtw_matrix.shape[0]):

        dtw_matrix.iloc[i,i] = np.nan
        edit_matrix.iloc[i,i] = np.nan


    dtw_matrix.to_csv("matrices/" + traj_data_sample + "_dtw_matrix_trajs.csv",index=False)
    edit_matrix.to_csv("matrices/" +  traj_data_sample + "_edit_matrix_trajs.csv",index=False)

    for index in dataset_traj_number_geo25["0"].index:
        for s in dataset_traj_number_geo25["0"].loc[index]:
            sensors_traj_sample.append(s.lower())

    sensors_traj_sample = list(set(sensors_traj_sample))

    sample_sensor_id_list  = []
    for index in tokenizer_df.index:
        if(tokenizer_df.sensor.loc[index].lower() in sensors_traj_sample):
            sample_sensor_id_list.append(index)

        elif(tokenizer_df.sensor.loc[index]=="[CSL]" or tokenizer_df.sensor.loc[index]=="[SEP]" or tokenizer_df.sensor.loc[index]=="[MASK]"):
            sample_sensor_id_list.append(index)
            
    return query_index, sample_sensor_id_list, sensors_traj_sample, edit_matrix, dtw_matrix, cossine_matrix_2, cossine_matrix_1, dataset_traj_number_geo25, traj_number, dataset_traj_number, object_traj, objects_traj, dict_traj_id, lat_long_2_sample, traj_data_sample, out2


def sensor_matrix_read(embedding, cossine_matrix_sensors_1, cossine_matrix_sensors_2, euclidean_matrix_sensors, road_matrix_sensors, sample_sensor_id_list, lat_long_2, objects_loc, object_loc):
    
    for file in listdir("./matrices"):

        if((embedding in file) and not("sample" in file)):

            cossine_matrix_sensors_1 = pd.read_csv("matrices/" + embedding + "_cossine_matrix_sensors.csv")
            cossine_matrix_sensors_2 = pd.read_csv("matrices/" + embedding + "_cm_sensors.csv")
            euclidean_matrix_sensors = pd.read_csv("matrices/" + embedding + "_euclidean_matrix_sensors.csv")

            cossine_matrix_sensors_1.index = [int(i) for i in cossine_matrix_sensors_1.columns]
            cossine_matrix_sensors_2.index = [int(i) for i in cossine_matrix_sensors_2.columns]
            cossine_matrix_sensors_2.columns = [int(i) for i in cossine_matrix_sensors_2.columns]
            euclidean_matrix_sensors.index = [int(i) for i in euclidean_matrix_sensors.columns]

            columns_sensors = np.array(sorted(np.intersect1d(euclidean_matrix_sensors.columns.values,np.intersect1d(cossine_matrix_sensors_1.columns.values, road_matrix_sensors.columns.values)),key=lambda x : int(x)))
            index_sensors = np.intersect1d(euclidean_matrix_sensors.index.values,np.intersect1d(cossine_matrix_sensors_1.index.values, road_matrix_sensors.index.values))

            road_matrix_sensors = road_matrix_sensors.loc[index_sensors,columns_sensors]

            if(101 in sample_sensor_id_list):
                sample_sensor_id_list.remove(101)
            if(102 in sample_sensor_id_list):
                sample_sensor_id_list.remove(102)
            if(103 in sample_sensor_id_list):
                sample_sensor_id_list.remove(103)

            lat_long_2["trajectory_id"] = lat_long_2.index

            objects_loc = sample_sensor_id_list
            object_loc = objects_loc[0]

            
            return False, cossine_matrix_sensors_1, cossine_matrix_sensors_2, euclidean_matrix_sensors, road_matrix_sensors, sample_sensor_id_list, lat_long_2, objects_loc, object_loc
        
    return True, cossine_matrix_sensors_1, cossine_matrix_sensors_2, euclidean_matrix_sensors, road_matrix_sensors, sample_sensor_id_list, lat_long_2, objects_loc, object_loc
    
def sensor_matrix_creation(embedding, embedding_matrix, tokenizer_df, euclidean_matrix_sensors, cossine_matrix_sensors_1, cossine_matrix_sensors_2, lat_long_1, lat_long_2, sample_sensor_id_list, objects_loc, object_loc):
    
    dist_matrix_coss_sensor = np.zeros((len(sample_sensor_id_list),len(sample_sensor_id_list)))
    eucl_distance = np.zeros((len(sample_sensor_id_list),len(sample_sensor_id_list)))

    for i in range(len(sample_sensor_id_list)):
        for j in range(len(sample_sensor_id_list)):
            dist_matrix_coss_sensor[i,j] = nltk.cluster.cosine_distance(np.array(embedding_matrix)[sample_sensor_id_list[i]-1],np.array(embedding_matrix)[sample_sensor_id_list[j]-1])


    tokenizer_df_sample = tokenizer_df.loc[sample_sensor_id_list]


    lat_values = []


    for i in tokenizer_df_sample.index:

        if(i not in [101,102,103] ):
            lat_values.append(lat_long_1.loc[tokenizer_df_sample[tokenizer_df_sample["id"] == i]["sensor"].apply(lambda x : x.lower())]["lat"].values[0])

        else:

            lat_values.append("n")

    lon_values = []

    for i in tokenizer_df_sample.index:

        if(i not in [101,102,103]):

            lon_values.append(lat_long_1.loc[tokenizer_df_sample[tokenizer_df_sample["id"] == i]["sensor"].apply(lambda x : x.lower())]["lon"].values[0])

        else:

            lon_values.append("n")    


    for i in range(len(lat_values)):
        for j in range(len(lat_values)):
            if((lat_values[i] != "n") and (lat_values[j] != "n") ):     
                eucl_distance[i,j] = distances.euclidean_distance_in_meters(lat_values[i],lon_values[i],lat_values[j],lon_values[j])



    cossine_matrix_sensors_1 = pd.DataFrame(dist_matrix_coss_sensor)

    euclidean_matrix_sensors = pd.DataFrame(eucl_distance)

    for i in cossine_matrix_sensors_1.index:

        cossine_matrix_sensors_1.iloc[i,i] = np.nan 

    for i in euclidean_matrix_sensors.index:

        euclidean_matrix_sensors.iloc[i,i] = np.nan

    cossine_matrix_sensors_1.columns = [str(i) for i in sample_sensor_id_list]
    euclidean_matrix_sensors.columns = [str(i) for i in sample_sensor_id_list]
    cossine_matrix_sensors_1.index = sample_sensor_id_list
    euclidean_matrix_sensors.index = sample_sensor_id_list

    columns_sensors = np.array(sorted(np.intersect1d(euclidean_matrix_sensors.columns.values,np.intersect1d(cossine_matrix_sensors_1.columns.values, road_matrix_sensors.columns.values)),key=lambda x : int(x)))
    index_sensors = np.intersect1d(euclidean_matrix_sensors.index.values,np.intersect1d(cossine_matrix_sensors_1.index.values, road_matrix_sensors.index.values))

    cossine_matrix_sensors_1 = cossine_matrix_sensors_1.loc[index_sensors,columns_sensors]

    euclidean_matrix_sensors = euclidean_matrix_sensors.loc[index_sensors,columns_sensors]

    road_matrix_sensors = road_matrix_sensors.loc[index_sensors,columns_sensors]

    cossine_matrix_sensors_1.to_csv("matrices/" + embedding + "_cossine_matrix_sensors.csv",index=False)
    euclidean_matrix_sensors.to_csv("matrices/" + embedding + "_euclidean_matrix_sensors.csv",index=False)

    cossine_matrix_sensors_2 = copy(cossine_matrix_sensors_1)
    cossine_matrix_sensors_2.columns = [int(i) for i in cossine_matrix_sensors_2.columns]
    cossine_matrix_sensors_2.to_csv("matrices/" + embedding + "_cm_sensors.csv",index=False)


    lat_long_2["trajectory_id"] = lat_long_2.index


    if(101 in sample_sensor_id_list):
        sample_sensor_id_list.remove(101)
    if(102 in sample_sensor_id_list):
        sample_sensor_id_list.remove(102)
    if(103 in sample_sensor_id_list):
        sample_sensor_id_list.remove(103)

    objects_loc = sample_sensor_id_list
    object_loc = objects_loc[0]
    
    return tokenizer_df, euclidean_matrix_sensors, cossine_matrix_sensors_1, cossine_matrix_sensors_2, lat_long_1, lat_long_2, sample_sensor_id_list, objects_loc, object_loc

            
def on_change_traj_data(change, query_index, cossine_matrix_1, cossine_matrix_2, dtw_matrix, edit_matrix, cossine_matrix_sensors_1, cossine_matrix_sensors_2, euclidean_matrix_sensors, road_matrix_sensors,
                       traj_data, traj_data_sample, tokenizer_df, sensors_traj_sample, embedding_matrix, embedding, lat_long_1, lat_long_2, lat_long_2_sample, dataset_traj_number, dataset_traj_number_geo25, 
                       traj_number, sample_sensor_id_list, object_loc, object_traj, objects_loc, objects_traj, dict_traj_id, select_trajectory_dropdown, select_location_dropdown, text_dataset, embedding_dictionary, out1, out2, out3):

        
        with out1:
            
            out1.clear_output()
            
            display(widgets.Label("Processing..."))
        
        
        traj_data = change.new
        validation = True
        validation_2 = True
        
        if(traj_data != ''):      
            
            t_id_list, traj_id_dict_all, lat_long_2 = data_traj_list_dict(lat_long_2, traj_data, tokenizer_df)
            
            validation, traj_number, traj_data_sample, lat_long_2_sample, dict_traj_id, objects_traj, object_traj, dataset_traj_number, dataset_traj_number_geo25, cossine_matrix_1, cossine_matrix_2, dtw_matrix, edit_matrix, sample_sensor_id_list, sensors_traj_sample, out2  =  \
            data_sample_traj_matrix_read(validation, traj_number, embedding, embedding_matrix, traj_data, traj_data_sample, lat_long_2_sample, dict_traj_id, objects_traj, object_traj,dataset_traj_number, dataset_traj_number_geo25, cossine_matrix_1, cossine_matrix_2, dtw_matrix, edit_matrix, traj_id_dict_all,
                                                                        sample_sensor_id_list, sensors_traj_sample, tokenizer_df, select_trajectory_dropdown, out2)
            
            if(validation):
        
                query_index, traj_number, sample_sensor_id_list, sensors_traj_sample, edit_matrix, dtw_matrix, cossine_matrix_2, cossine_matrix_1, dataset_traj_number_geo25, traj_number, dataset_traj_number, object_traj, objects_traj, dict_traj_id, lat_long_2_sample, traj_data_sample, out2  =  \
                data_sample_traj_matrix_creation(sensors_traj_sample, embedding, embedding_matrix, edit_matrix, dtw_matrix, cossine_matrix_1, dataset_traj_number_geo25, traj_number, dataset_traj_number, object_traj, objects_traj, dict_traj_id, lat_long_2_sample, lat_long_2, 
                                                 traj_data, t_id_list, traj_id_dict_all,traj_data_sample, tokenizer_df, select_trajectory_dropdown, out2)
            
            
            validation_2, cossine_matrix_sensors_1, cossine_matrix_sensors_2, euclidean_matrix_sensors, road_matrix_sensors, sample_sensor_id_list, lat_long_2, objects_loc, object_loc = \
            sensor_matrix_read(embedding, cossine_matrix_sensors_1, cossine_matrix_sensors_2, euclidean_matrix_sensors, road_matrix_sensors, sample_sensor_id_list, lat_long_2, objects_loc, object_loc)
            
            if(validation_2):
                
                tokenizer_df, euclidean_matrix_sensors, cossine_matrix_sensors_1, cossine_matrix_sensors_2, lat_long_1, lat_long_2, sample_sensor_id_list, objects_loc, object_loc = sensor_matrix_creation(embedding, embedding_matrix,tokenizer_df, euclidean_matrix_sensors, cossine_matrix_sensors_1, 
                                                                                                                                                                        cossine_matrix_sensors_2, lat_long_1, lat_long_2, sample_sensor_id_list, objects_loc, object_loc)
               
            
            with out3:
                
                out3.clear_output()
                select_location_dropdown.options = objects_loc
                select_location_dropdown.value = object_loc
                display(select_location_dropdown)
            
        else:
            
            pass
            
       
        with out1:
        
            text_dataset.options = [''] if(not(embedding in embedding_dictionary.keys())) else embedding_dictionary[embedding] + ['']
            text_dataset.value = ''
        
            out1.clear_output()
            
            display(text_dataset)
        

def traj_plot_mrr(b,cossine_matrix,edit_matrix,dtw_matrix,out):
        
    
        with out:
            
            out.layout = widgets.Layout(width = "450px",border='solid 2.0px white',margin='0px 5px 5px 5px',padding='2px 2px 2px 2px')
            out.clear_output()
            display(widgets.Label("Processing..."))


            mrrdtw=[]
            for dtw in range(5,100,5): 
                mrr_ = mean_reciprocal_rank_filter(cossine_matrix.loc[dtw_matrix.index.values,:], dtw_matrix , dtw, dtw+1,2)
                mrrdtw.append(mrr_)
                
            

            mrredit=[]
            for d in range(5,100,5): 
                mrr_ = mean_reciprocal_rank_filter(cossine_matrix, edit_matrix, d, d+1,2)
                mrredit.append(mrr_)
                
            

            plt.plot( range(5,100,5), mrrdtw, '-*')

            plt.plot( range(5,100,5), mrredit, '-o')

            plt.legend(['DTW', 'Edit distance'])
            plt.xlabel('distance')
            plt.ylabel('mrr')
            plt.title('Mean Reciprocal Ranking vs Maximal Distance')
            
            out.clear_output()
            
            plt.show()
            
            out.layout = widgets.Layout(width = "450px",border='solid 2.0px white',margin='0px 5px 5px 5px',padding='2px 2px 2px 2px')
            
def loc_mrr_plot(b,cossine_matrix,euclidean_matrix,road_matrix,out):
        
        with out:
            
            out.layout = widgets.Layout(width = "450px",border='solid 2.0px white',margin='0px 5px 5px 5px',padding='2px 2px 2px 2px')
            out.clear_output()
            display(widgets.Label("Processing..."))
            
            mrre=[]
            for ed in range(500,10001,500):
                mrr_ = mean_reciprocal_rank_filter(cossine_matrix, euclidean_matrix, ed,ed+1,2)
                mrre.append(mrr_)


            mrr=[]
            for rd in range(500,10001,500):
                mrr_ = mean_reciprocal_rank_filter(cossine_matrix, road_matrix, rd,rd+1,2)
                mrr.append(mrr_)

            plt.plot( range(500,10001,500), mrr, '-*')

            plt.plot( range(500,10001,500), mrre, '-o')

            plt.legend(['Road Distance', 'Euclidean Euclidean'])
            plt.xlabel('distance (meters)')
            plt.ylabel('mrr')
            plt.title('Mean Reciprocal Ranking vs Range Distance')
            plt.savefig('mrr_correct',format='pdf')
            
            
            out.clear_output()
            
            plt.show()
            
            out.layout = widgets.Layout(width = "450px",border='solid 2.0px white',margin='0px 5px 5px 5px',padding='2px 2px 2px 2px')
            
            
def traj_topk_plot(b, dataset_traj_number, object_traj, topk_traj, lat_long_sample, traj_number, dict_traj_id, cossine_matrix,out):
        
        
        with out:
            
            out.clear_output()
            
            display(widgets.Label("Processing..."))
           
            traj_matrix_index = dataset_traj_number[dataset_traj_number["trajectory_number"] == str(object_traj)].index[0]
            k = int(topk_traj)
            
            cossine_matrix = cossine_matrix.to_numpy() 
            
            neighborhood_topk = []
            for i in range(cossine_matrix.shape[0]):
                neighborhood_topk.append((cossine_matrix[i][traj_matrix_index], i))

            neighborhood_topk_index = sorted(neighborhood_topk)
            

            neighborhood_topk_index = neighborhood_topk_index[0:k+1]
            
           
            
            trajectories = [dataset_traj_number["trajectory_number"].loc[nb[1]] for nb in neighborhood_topk_index]
             

            trajectories_id = list(traj_number["0"].loc[[tj for tj in trajectories]])

            trajectories_df = pd.concat([filters.by_label(lat_long_sample, value = tj, label_name = "trajectory_id") for tj in trajectories_id],axis=0)
            
            trajectories_df['trajectory_id'] = trajectories_df['trajectory_id'].apply(lambda x : "Trajectory : "+str(dict_traj_id[x]))


            move_df = MoveDataFrame(data= trajectories_df, latitude="lat", longitude="lon", datetime="time", traj_id='trajectory_id')
            
            out.clear_output()
            
            display(f.plot_trajectories(move_df))
                   
def loc_topk_plot(b, cossine_matrix_sensors, lat_long, object_loc, topk_loc, sample_sensor_id_list, tokenizer_df, out):

    with out:

        out.clear_output()

        neighborhood_topk_s = []

        for index in cossine_matrix_sensors.index:
            cossine_matrix_sensors.loc[index][index] = 0.0

        cossine_matrix_s = cossine_matrix_sensors.to_numpy() 

        for i in range(cossine_matrix_s.shape[0]):
            neighborhood_topk_s.append((cossine_matrix_s[i][sample_sensor_id_list.index(int(object_loc))], sample_sensor_id_list[i])) 

        neighborhood_topk_s_index = sorted(neighborhood_topk_s)

        neighborhood_topk_s_index = neighborhood_topk_s_index[0:int(topk_loc)+1]


        closest_labels = tokenizer_df.loc[[n[1] for n in neighborhood_topk_s_index], :]
        aux_cl = closest_labels['sensor'].apply(lambda x : x.lower())

        closest_sensor = lat_long.loc[aux_cl]
        closest_sensor['id'] = ["Location: "+str(n[1]) for n in neighborhood_topk_s_index]
        closest_sensor['datetime'] = 0.0


        map = plot_trajectories(closest_sensor)

        out.clear_output()

        display(plot_points(closest_sensor, user_point='gray', base_map=map))