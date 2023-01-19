import pandas as pd
from IPython.display import display
import ipywidgets as widgets

from pymove import MoveDataFrame
import folium
from pymove.visualization import folium as f
from pymove import filters
from random import sample 


def on_change_dataset(change, dataset):

    dataset = change.new

def on_change_sampling(change, sampling):

    sampling = change.new


def show_statistics(b, dataset, out):

    with out:

        DF = pd.read_csv("data/"+ dataset +".csv")
        
        if(len(DF) > 0):
        
            out.clear_output()
            
            display(widgets.Label("Processing..."))
            
            number_of_trajectories = len(set(DF["trajectory_id"]))
            length_trajectories = [len(DF[DF["trajectory_id"] == tj]) for tj in set(DF["trajectory_id"])]
            maximum_length_of_trajectories = max(length_trajectories)
            minimum_length_of_trajectories = min(length_trajectories)
            average_length_of_trajectories = int(sum(length_trajectories)/len(length_trajectories))

            df1 = ""
    
            if("location_label" in list(DF.columns)):
                
                number_of_distinct_locations = len(set(DF["location_label"]))
                df1 = pd.DataFrame({" ":[number_of_trajectories,maximum_length_of_trajectories,minimum_length_of_trajectories,average_length_of_trajectories,
                                number_of_distinct_locations]}, index = ["number_of_trajectories","maximum_length_of_trajectories","minimum_length_of_trajectories",
                                                                        "average_length_of_trajectories","number_of_distinct_locations"])
            else:
                df1 = pd.DataFrame({" ":[number_of_trajectories,maximum_length_of_trajectories,minimum_length_of_trajectories,average_length_of_trajectories]}, 
                                    index = ["number_of_trajectories","maximum_length_of_trajectories","minimum_length_of_trajectories",
                                                                        "average_length_of_trajectories"])

            out.clear_output()
            display(df1)

        else:
            out.clear_output()
            display(widgets.Label("Dataset Vazio !"))

def plot_sampling(b, sampling, dataset, out):

    with out:

        DF = pd.read_csv("data/"+ dataset +".csv")
    
        if(len(DF) > 0):

            out.clear_output()
            
            display(widgets.Label("Processing..."))
            
            set_traj = set(DF["trajectory_id"])

            Sampling_Traj = sample(set_traj,int(sampling))
            DataFrame = pd.concat([filters.by_label(DF, value = tj, label_name = "trajectory_id") for tj in Sampling_Traj],axis=0)
    

            move_df = MoveDataFrame(data=DataFrame, latitude="lat", longitude="lon", datetime="time", traj_id='trajectory_id')
            figure = folium.Figure(width=800, height=300) 

            out.clear_output()
            
            display(f.plot_trajectories(move_df).add_to(figure))

        else:

            out.clear_output()
            display(widgets.Label("Dataset Vazio !"))