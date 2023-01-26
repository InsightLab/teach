import pandas as pd
from IPython.display import display
import ipywidgets as widgets

from pymove import MoveDataFrame
import folium
from pymove.visualization import folium as f
from pymove import filters
from os import listdir
from random import sample 

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from utils.output import *

class View_Trajectory_Data_Output:


    def __init__(self):

        self.dataset_list = [dataframe.split(".")[0] for dataframe in listdir("data/")]               
        self.dataset  = self.dataset_list[0]

        self.spc = widgets.Label("")
        self.spc1 = widgets.Label("",layout=widgets.Layout(width="1px",height="1px"))

        self.out_df = out(widgets.Label(""))
        self.out_plot = out(widgets.Label(""))
        self.out_df.layout = widgets.Layout(border='solid 1.5px white',margin='0px 10px 10px 10px',padding='5px 5px 5px 5px')

        self.out_plot.layout = widgets.Layout(border='solid 2.0px white',margin='0px 10px 10px 10px',padding='5px 5px 5px 5px')
    
        self.out_statistics = widgets.Output()
        self.out_statistics.layout = widgets.Layout(border='solid 1.5px white',margin='10px 70px 10px 0px', padding='5px 5px 5px 5px')

        self.out_sampling = widgets.Output()
        self.out_sampling.layout = widgets.Layout(border='solid 1.5px white',margin='20px 10px 10px 0px', padding='5px 5px 5px 5px')

        self.select_data_dropdown = widgets.Dropdown(description='',layout=widgets.Layout(width="80px"),options=self.dataset_list)
        self.select_data_dropdown.observe(self.on_change_dataset,  names='value')
        self.select_data_dropdown_output = out(self.select_data_dropdown)

        self.sampling = 10

        with self.out_statistics:
            
            self.select_data = widgets.Label("Select Data")
            self.show_statistics_button = widgets.Button(description="Show Statistics", layout=widgets.Layout(width="120px"))
            self.show_statistics_button.on_click(self.show_statistics)
            self.show_statistics_button.style.button_color = "lightgray"

            aux = widgets.VBox(children=[widgets.HBox([self.spc,self.spc,self.select_data]),widgets.HBox(children=[self.spc, self.select_data_dropdown_output, self.spc,widgets.VBox([self.spc1,self.show_statistics_button])])])
            self.statistics_box = widgets.VBox(children=[aux,widgets.Label("",layout=widgets.Layout(width="19%")),self.out_df])

            display(self.statistics_box) 

        with self.out_sampling: 
            
            self.sampling_value = widgets.Label("Sampling")
            self.sampling_value_text  = widgets.Text(description='',layout=widgets.widgets.Layout(width="80px"))
            self.sampling_value_text.observe(self.on_change_sampling, names = 'value')

            self.sampling_button = widgets.Button(description="Plot Trajectories",layout=widgets.Layout(width="140px"))
            self.sampling_button.style.button_color = "lightgray"
            self.sampling_button.on_click(self.plot_sampling)

            aux = widgets.VBox(children=[self.sampling_value,widgets.HBox(children=[self.sampling_value_text,self.sampling_button])])
            self.sampling_box = widgets.VBox(children=[aux,self.spc,self.out_plot])

            display(self.sampling_box)

        self.view_trajectory_data_box = widgets.HBox(children=[self.out_statistics,self.out_sampling])

        


    def on_change_dataset(self,change):

        self.dataset = change.new

    def on_change_sampling(self,change):

        self.sampling = change.new


    def show_statistics(self, show_statistics_button):

        with self.out_df:

            DF = pd.read_csv("data/"+ self.dataset +".csv")
            
            if(len(DF) > 0):
            
                self.out_df.clear_output()
                
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

                self.out_df.clear_output()
                display(df1)

            else:
                self.out_df.clear_output()
                display(widgets.Label("Dataset Vazio !"))

    def plot_sampling(self, sampling_button):

        with self.out_plot:

            DF = pd.read_csv("data/"+ self.dataset +".csv")
        
            if(len(DF) > 0):

                self.out_plot.clear_output()
                
                display(widgets.Label("Processing..."))
                
                set_traj = set(DF["trajectory_id"])

                Sampling_Traj = sample(set_traj,int(self.sampling))
                DataFrame = pd.concat([filters.by_label(DF, value = tj, label_name = "trajectory_id") for tj in Sampling_Traj],axis=0)
        

                move_df = MoveDataFrame(data=DataFrame, latitude="lat", longitude="lon", datetime="time", traj_id='trajectory_id')
                figure = folium.Figure(width=800, height=300) 

                self.out_plot.clear_output()
                
                display(f.plot_trajectories(move_df).add_to(figure))

            else:

                self.out_plot.clear_output()
                display(widgets.Label("Dataset Vazio !"))