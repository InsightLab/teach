import pathlib
import sys
import functools
import pickle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dashboards_functions.extrinsic_evaluation import *
from dashboards_functions.intrinsic_evaluation import *
from dashboards_functions.view_trajectory_data import *
from dashboards_functions.upload import *

def out(ds):
    
    out = widgets.Output()
    with out:
        display(ds)
    return out

data_list  = [dt.split(".")[0] for dt in listdir("data/")]

teach_variable_states = { "data_name": "",
                        
                          "data_list": data_list

                        }

with open('teach_variable_states.pickle', 'wb') as handle:
    pickle.dump(teach_variable_states, handle)



embedding = "" 
tokenizer_df = ""

Emb_list = list(pd.read_csv("Emb.csv").Emb)
Emb_df = pd.read_csv("Emb#Data.csv") 
Emb_dict = { emb: Emb_df[Emb_df["Emb"]==emb].Data.values[0].split("%") for emb in list(Emb_df.Emb)}

Model_df = pd.read_csv("Model#Data.csv")
Model_dict = {md:dt.split("%")[0] for md,dt in zip(Model_df.Model, Model_df.Data)}
models_names = list(Model_df.Model)

DatasetsList = [dataframe.split(".")[0] for dataframe in listdir("data/")]               
dataset_2  = DatasetsList[0]


select_data_dropdown = widgets.Dropdown(description='',layout=widgets.Layout(width="80px"),options=DatasetsList)
select_data_dropdown.observe(functools.partial(on_change_dataset, dataset= dataset_2), names='value')
select_data_dropdown_out = out(select_data_dropdown)

text_dataset = widgets.Select(description="Select Data:",layout=widgets.Layout(width="380px",height="180px"), options= [''] if(not(Emb_list[0] in Emb_dict.keys())) else Emb_dict[Emb_list[0]] + [''],value='')
text_dataset_out = out(text_dataset)

model_text = widgets.Label("Select Embedding")
model_choice = widgets.Dropdown(description='',layout=widgets.Layout(width="150px"),options=Emb_list,value=Emb_list[0])
model_choice.observe(functools.partial(on_change_embedding_choice, embedding= embedding, emb_dict= Emb_dict, tokenizer_df= tokenizer_df, text_dataset= text_dataset, out= text_dataset_out),names="value")
model_choice_out = out(model_choice)

select_mutiples_models = widgets.SelectMultiple(options=models_names + [''],value=[''] ,description='Models:', layout= widgets.Layout(width="420px",height="140px"),disabled=False)  
select_mutiples_models_output = out(select_mutiples_models)




def upload_output():

    data_upload = ""
    embedding_multi_select_upload = ""
    model_select= ""

    data_upload_output = ""
    embedding_multi_select_upload_output = ""
    model_select_output = ""
    
    model_multi_select_upload= ""
    model_multi_select_upload_output= ""


    model_name = ""
    embedding_name = ""
    
    model_multi_select_2 = ""
    embedding_select = ""
    embedding_select_output = ""
    embedding_data_select_multiple = ""
    embedding_data_select_multiple_output = ""
    model_data_select_multiple= ""
    model_data_select_multiple_output= ""

    spc1 = widgets.Label("",layout=widgets.Layout(width="9%"))
    spc2 = widgets.Label("",layout=widgets.Layout(width="6%"))

    
    
    
    model_list = [model.split(".")[0] for model in listdir("models/")]
    embedding_list   = [eb.split(".")[0] for eb in listdir("embeddings/")]

    teach_variable_states = {}

    with open('teach_variable_states.pickle', 'rb') as handle:
        teach_variable_states = pickle.load(handle)

    data_list = teach_variable_states["data_list"]

   
    
    if("" in data_list):
        data_list.remove("")  
        
    if("" in data_list):
        data_list.remove("")
        
    if("" in embedding_list):
        embedding_list.remove("")


    model_data_multi_select_dict_show = {mdl:mdl for mdl in data_list}
    data_multi_select_dict_show = {dl:dl for dl in data_list}

    embedding_upload_button = widgets.FileUpload(acept=".csv",multiple=True,layout=widgets.Layout(width="150px"))
    embedding_upload_button.observe(functools.partial(upload_embedding, model_choice= model_choice, embedding_dictonary= Emb_dict, embedding_dataframe= Emb_df, embedding_select= embedding_select, embedding_list_2= Emb_list, embedding_list_1= embedding_list, 
                                    embedding_multi_select_up= embedding_multi_select_upload, out1= embedding_multi_select_upload_output, out2= embedding_select_output, out3= model_choice_out), names="value")
    
    
    remove_embedding_button = widgets.Button(description="Remove", layout=widgets.Layout(width="150px"))
    remove_embedding_button.on_click(functools.partial(remove_embedding,  model_choice= model_choice, embedding_list_2= embedding_list, embedding_select= embedding_select, embedding_multi_select_up= embedding_multi_select_upload, 
                                                        embedding_list_1= Emb_list, embedding_dataframe= Emb_df, embedding_dictionary= Emb_dict, out1= embedding_select_output, out2= embedding_multi_select_upload_output, out3= model_choice_out))
    
    
    rename_embedding_button = widgets.Button(description="Rename", layout=widgets.Layout(width="90px"))
    rename_embedding_button.style.button_color = "lightgray"
    rename_embedding_button.on_click(functools.partial(rename_embedding, embedding_name= embedding_name, embedding_list_1= embedding_list , embedding_select= embedding_select, embedding_multi_select_up= embedding_multi_select_upload, model_choice= model_choice, 
                                    on_change_emb_ml= on_change_emb_ml, embedding_list_2= Emb_list, embedding_dataframe= Emb_df, embedding_dictionary= Emb_dict, out1= embedding_multi_select_upload_output, out2= embedding_select_output, out3= model_choice_out))
    
    embedding_name_label = widgets.Text(description='',layout=widgets.widgets.Layout(width="250px"))
    embedding_name_label.observe(functools.partial(on_change_emb_name, embedding_name= embedding_name), names = 'value')
    
    
    if(len(embedding_list)>0):
        embedding_multi_select_upload = widgets.SelectMultiple(options=embedding_list,description='Emb:', value=[embedding_list[0]],layout=widgets.Layout(width="380px",height="180px"))

    else:       
        embedding_multi_select_upload = widgets.SelectMultiple(options=embedding_list,description='Emb:', value=[],layout=widgets.Layout(width="380px",height="180px"))
        
    embedding_multi_select_upload_output = out(embedding_multi_select_upload)
    
    embedding_upload_box = widgets.HBox(children=[widgets.VBox([embedding_upload_button,remove_embedding_button,spc1]),spc1,widgets.VBox([embedding_multi_select_upload_output,spc2, widgets.HBox([spc1,spc1,spc2,embedding_name_label,rename_embedding_button])])])
    
    
    embedding_data_select_multiple = widgets.SelectMultiple(options=[data_multi_select_dict_show[dl] for dl in data_list],description='Data:', layout=widgets.Layout(width="380px",height="180px"))
    embedding_data_select_multiple.observe(functools.partial(on_change_emb_data_ml, embedding_data_multi_select= embedding_data_select_multiple, data_list= data_list, data_multi_select_dict_show= data_multi_select_dict_show, 
                                                                embedding_select= embedding_select), names = 'value')
    embedding_data_select_multiple_output = out(embedding_data_select_multiple)
    
    
    embedding_select = widgets.Select(options=embedding_list + [''],description='Emb:',value='',layout=widgets.Layout(width="380px",height="180px"))
    embedding_select.observe(functools.partial(on_change_emb_ml, embedding_data_multi_select= embedding_data_select_multiple, embedding_data_multi_select_output= embedding_data_select_multiple_output, data_list= data_list), names = 'value')
    embedding_select_output = out(embedding_select)
    
    
    
    embedding_link_button = widgets.Button(description="Link", layout=widgets.Layout(width="80px"))
    embedding_link_button.style.button_color = "lightgray"
    embedding_link_button.on_click(functools.partial(link_embedding, model_choice= model_choice, text_dataset= text_dataset, embedding_select= embedding_select, data_list= data_list, embedding_list= Emb_list, embedding_dataframe= Emb_df, 
                                embedding_dictionary= Emb_dict, emb_data_multi_select= embedding_data_select_multiple, data_multi_select_dict_show= data_multi_select_dict_show, out1= embedding_data_select_multiple_output, out2= model_choice_out, out3= text_dataset_out))

    
    embedding_unlink_button = widgets.Button(description="Ulink", layout=widgets.Layout(width="80px"))
    embedding_unlink_button.style.button_color = "lightgray"
    embedding_unlink_button.on_click(functools.partial(unlink_embedding, model_choice= model_choice, text_dataset= text_dataset,embedding_select= embedding_select, 
                         emb_data_multi_select= embedding_data_select_multiple, data_list= data_list, embedding_dataframe= Emb_df, embedding_list= Emb_list, data_multi_select_dict_show= data_multi_select_dict_show, 
                         embedding_dictonary= Emb_dict, out1= embedding_data_select_multiple_output, out2= model_choice_out, out3= text_dataset_out))
    
    
    
    embedding_link_box = widgets.VBox([widgets.HBox([embedding_link_button,widgets.Label("",layout=widgets.Layout(width="1%")),embedding_unlink_button]), spc2,widgets.HBox([embedding_select_output,spc2, embedding_data_select_multiple_output])])


    model_upload_button = widgets.FileUpload(acept=".h5",multiple=True,layout=widgets.Layout(width="150px"))
    model_upload_button.observe(functools.partial(upload_model, model_multi_select_up= model_multi_select_upload, model_list= model_list, model_multi_select= model_select, out1= model_multi_select_upload_output, out2= model_select_output), names="value")

    model_remove_button = widgets.Button(description="Remove", layout=widgets.Layout(width="150px"))
    model_remove_button.on_click(functools.partial(remove_model, models_names, model_list= model_list, model_multi_select= model_select,  w_sm= select_mutiples_models, model_multi_select_up= model_multi_select_upload, model_dataframe= Model_df, model_dictonary= Model_dict, 
                        out1= model_multi_select_upload_output, out2= model_select_output, out3= select_mutiples_models))
    
    
    model_rename_button = widgets.Button(description="Rename", layout=widgets.Layout(width="90px"))
    model_rename_button.style.button_color = "lightgray"
    model_rename_button.on_click(functools.partial(rename_model, model_name= model_name, models_names= models_names, model_list= model_list, model_multi_select= model_select, model_multi_select_up= model_multi_select_upload, model_dataframe= Model_df, 
                                        model_dictionary= Model_dict, out1= model_multi_select_upload_output, out2= model_select_output, out3=  select_mutiples_models))
    
    model_name_label = widgets.Text(description='',layout=widgets.widgets.Layout(width="250px"))
    model_name_label.observe(functools.partial(on_change_model_name, model_name= model_name), names = 'value')
    
    
    
    if(len(model_list)>0):
    
        model_multi_select_upload = widgets.SelectMultiple(options=model_list,description='Models:', value=[model_list[0]],layout=widgets.Layout(width="380px",height="180px"))

    else:
        
         model_multi_select_upload = widgets.SelectMultiple(options=model_list,description='Models:', value=[],layout=widgets.Layout(width="380px",height="180px"))
    
    model_multi_select_upload_output = out(model_multi_select_upload)
        
    
    model_upload_box = widgets.HBox(children=[widgets.VBox([model_upload_button, model_remove_button,spc1]), spc1,widgets.VBox([model_multi_select_upload_output,spc2, widgets.HBox([spc1,spc1,spc2,model_name_label,model_rename_button])])])
    
    
    model_data_select_multiple = widgets.SelectMultiple(options=[model_data_multi_select_dict_show[mdl] for mdl in data_list],description='Data:',layout=widgets.Layout(width="380px",height="180px"))
    model_data_select_multiple.observe(functools.partial(on_change_model_data_ml, model_data_multi_select= model_data_select_multiple), names = 'value')
    model_data_select_multiple_output = out(model_data_select_multiple)
    
    model_select = widgets.Select(options=model_list + [''],description='Models:',value='',layout=widgets.Layout(width="380px",height="180px"))
    model_select.observe(functools.partial(on_change_model_ml, model_data_multi_select= model_data_select_multiple, model_data_multi_select_output= model_data_select_multiple_output, model_data_multi_select_dict_show= model_data_multi_select_dict_show,
                                                     model_multi_select= model_multi_select_2, data_list= data_list), names = 'value')
    model_select_output = out(model_select)
    
    model_link_button = widgets.Button(description="Link", layout=widgets.Layout(width="80px"))
    model_link_button.style.button_color = "lightgray"
    model_link_button.on_click(functools.partial(link_model,  models_names= models_names, data_list= data_list, model_dictionary= Model_dict, model_dataframe= Model_df, select_mutiples_models= select_mutiples_models,model_multi_select= model_select, 
                                                model_data_multi_select= model_data_select_multiple, model_data_multi_select_dict_show= model_data_multi_select_dict_show, out1= model_data_select_multiple_output, out2= select_mutiples_models))
    
    model_unlink_button = widgets.Button(description="Ulink", layout=widgets.Layout(width="80px"))
    model_unlink_button.style.button_color = "lightgray"
    model_unlink_button.on_click(functools.partial(unlink_model, models_names= models_names, select_mutiples_models= select_mutiples_models, data_list= data_list, model_data_multi_select= model_data_select_multiple, model_multi_select= model_select, model_dataframe= Model_df, 
                        model_data_multi_select_dict_show= model_data_multi_select_dict_show, model_dictionary= Model_dict, out1= model_data_select_multiple_output, out2=  select_mutiples_models))
    
    model_link_box = widgets.VBox([widgets.HBox([model_link_button,widgets.Label("",layout=widgets.Layout(width="1%")),model_unlink_button]), spc2,widgets.HBox([model_select_output,spc2,model_data_select_multiple_output])])


    data_upload_button = widgets.FileUpload(acept=".csv",multiple=True,layout=widgets.Layout(width="150px"))
    data_upload_button.observe(functools.partial(upload_data, data_multi_select_dict_show= data_multi_select_dict_show, on_change_dataset= on_change_dataset, data_list= data_list, data_up= data_upload, emb_data_multi_select= embedding_data_select_multiple, 
                                              model_data_multi_select= model_data_select_multiple, model_data_multi_select_dict_show= model_data_multi_select_dict_show,
                                              out1= data_upload_output, out2= embedding_data_select_multiple_output, out3= model_data_select_multiple_output, out4= select_data_dropdown_out), names="value")
    
    data_remove_button = widgets.Button(description="Remove", layout=widgets.Layout(width="150px"))
    data_remove_button.on_click(functools.partial(remove_data, model_choice= model_choice, text_dataset= text_dataset, on_change_dataset= on_change_dataset, on_change_model_data_ml= on_change_model_data_ml, select_mutiples_models= select_mutiples_models, data_list= data_list, 
                                          data_up= data_upload, models_names= models_names, embedding_list= Emb_list, model_dataframe= Model_df, embedding_dataframe= Emb_df, model_dictionary= Model_dict, model_data_multi_select_dic_show= model_data_multi_select_dict_show, 
                                          data_multi_select_dict_show= data_multi_select_dict_show, embedding_dictionary= Emb_dict, out1= data_upload_output, out2= embedding_data_select_multiple_output, out3= model_data_select_multiple_output, out4= select_mutiples_models, 
                                          out5= select_data_dropdown_out, out6= model_choice_out, out7= text_dataset_out))
    
    
    if(len(data_list)>0):
    
         data_upload = widgets.SelectMultiple(options=data_list,description='Data:', value=[data_list[0]],layout=widgets.Layout(width="380px",height="180px"))
    else:
        
         data_upload = widgets.SelectMultiple(options=data_list,description='Data:', value=[],layout=widgets.Layout(width="380px",height="180px"))
    
    data_upload_output = out(data_upload)

    data_name_label = widgets.Text(description='',layout=widgets.widgets.Layout(width="250px"))
    data_name_label.observe(functools.partial(on_change_data_name), names = 'value')

    
    data_rename_button = widgets.Button(description="Rename", layout=widgets.Layout(width="90px"))
    data_rename_button.style.button_color = "lightgray"
    data_rename_button.on_click(functools.partial(rename_data, model_choice= model_choice, text_dataset= text_dataset, on_change_dataset= on_change_dataset, on_change_model_data_ml= on_change_model_data_ml, on_change_emb_data_ml= on_change_emb_data_ml,
                                        models_names= models_names, data_list= data_list, data_up= data_upload, select_mutiples_models= select_mutiples_models, model_data_multi_select_dict_show= model_data_multi_select_dict_show, data_multi_select_dict_show= data_multi_select_dict_show, 
                                        model_dataframe= Model_df, model_dictionary= Model_dict, embedding_list= Emb_list, embedding_dataframe= Emb_df, embedding_dictionary= Emb_dict, out1= data_upload_output, out2= embedding_data_select_multiple_output, 
                                        out3= model_data_select_multiple_output, out4= select_mutiples_models_output, out5= select_data_dropdown_out, out6= model_choice_out, out7= text_dataset_out))
    
    
    
    
    data_upload_box = widgets.HBox(children=[widgets.VBox([data_upload_button, data_remove_button, spc1]), spc1,widgets.VBox([data_upload_output,spc2, widgets.HBox([spc1,spc1,spc2, data_name_label, data_rename_button])])])
    
    
    
    tab_embedding = widgets.Tab()
    tab_embedding.children = [embedding_upload_box ,embedding_link_box]
    tab_embedding.set_title(0,'Upload')
    tab_embedding.set_title(1,'Data Linkage') 
    
    tab_model = widgets.Tab()
    tab_model.children = [model_upload_box,model_link_box]
    tab_model.set_title(0,'Upload')
    tab_model.set_title(1,'Data Linkage')
    
    accordion = widgets.Accordion(children=[data_upload_box,tab_embedding,tab_model],selected_index=None)
    
    accordion.set_title(0, 'Datasets Import')
    accordion.set_title(1, 'Embeddings Import')
    accordion.set_title(2, 'Models Import')

    display(accordion)

def intrinsic_evaluation_output(intrinsic_evaluation_output):

    display(widgets.Label("Processing..."))

    spc = widgets.Label("")
    spc1 = widgets.Label("",layout=widgets.Layout(width="9%"))
    
    traj_data = ""
    t_id_list = []
    traj_id_dict_all = {}
    traj_id_dict = {}
    traj_data_sample = ""

    LatLong_sample = ""
    LatLong   = ""
    LL = ""
    
    Dataset = ""
    embedding_matrix = ""
    Dataset_geo25 = ""
    Dict_Geo25 = {}

    Cossine_Matrix_sensors = ""
    CM_sensors = ""
    Eucli_Matrix_sensors = ""
    Road_Matrix_sensors = ""

    Cossine_Matrix= ""
    CM = ""
    DTW_Matrix = ""
    Edit_Matrix = ""
    
    Objects_traj = []
    obj_traj = ""
    Objects_loc = []
    obj_loc = ""
    
    query_index = ""
    
    Traj_Number_Traj_id = ""
    Dataset_geo25 = ""
    

    sensors_traj_sample = []
    sts_id = []

    topk_loc = -1
    topk_traj = -1


    if(len(list(Emb_df.Emb))==0):
        
        LL, tokenizer_df, embedding_matrix = embedding_not_linked(embedding_list= Emb_list)
        
    
    elif(len(list(Emb_df.Emb))>1):
    
        LL, LatLong, tokenizer_df, embedding_matrix, embedding, t_id_list, traj_id_dict_all, traj_data = embedding_linked(embedding_list= Emb_list, embedding_dict= Emb_dict)
        
        validation = True
        
        validation, query_index , traj_data_sample, LatLong_sample, traj_id_dict, Objects_traj, obj_traj, Dataset, Dataset_geo25, Dict_Geo25, Traj_Number_Traj_id, sts_id, Cossine_Matrix, CM, DTW_Matrix, Edit_Matrix = \
        data_sample_traj_matrix_first_read(traj_data, embedding, traj_id_dict_all, tokenizer_df, embedding_matrix)
        
        if(validation):
        
            query_index, traj_id_dict, LatLong_sample, Objects_traj, obj_traj, Dataset, Dataset_geo25, Dict_Geo25, Traj_Number_Traj_id, Cossine_Matrix, CM, DTW_Matrix, Edit_Matrix, sts_id  = \
            data_sample_traj_matrix_first_creation(embedding, traj_data, t_id_list, LatLong, traj_id_dict_all, tokenizer_df, embedding_matrix)

        
        if("Road_Matrix_sensors.csv" in listdir("./matrices")):
    
            Road_Matrix_sensors = pd.read_csv("matrices/Road_Matrix_sensors.csv")
            Road_Matrix_sensors.index = [int(i) for i in Road_Matrix_sensors.columns]
        
        else:
            
            Road_Matrix_sensors = road_matrix_creation(intrinsic_evaluation_output, tokenizer_df, sts_id, LL)


        validation_2 = True

        validation_2, Cossine_Matrix_sensors, CM_sensors, Eucli_Matrix_sensors, Road_Matrix_sensors, Objects_loc, obj_loc, sts_id, LatLong = sensor_matrix_first_read(embedding, Road_Matrix_sensors, sts_id, LatLong)
        
        if(validation_2):
            
            Cossine_Matrix, CM_sensors, Eucli_Matrix_sensors, Road_Matrix_sensors, LatLong, Objects_loc, obj_loc, sts_id = sensor_matrix_first_creation(embedding, Road_Matrix_sensors, tokenizer_df, sts_id, embedding_matrix, LatLong, Objects_loc, obj_loc)

    
    out_mrr_traj = out(widgets.Label(""))
    out_mrr_traj.layout = widgets.Layout(width = "450px", border='solid 2.0px white', margin='0px 5px 5px 5px', padding='2px 2px 2px 2px')
    
    out_mrr_loc = out(widgets.Label(""))
    out_mrr_loc.layout = widgets.Layout(width = "450px", border='solid 2.0px white', margin='0px 5px 5px 5px', padding='2px 2px 2px 2px')
    
    out_topk_traj = out(widgets.Label(""))
    out_topk_traj.layout = widgets.Layout(width = "825px", border='solid 2.0px white', margin='0px 10px 10px 10px', padding='5px 5px 5px 5px')
    
    out_topk_loc = out(widgets.Label(""))
    out_topk_loc.layout = widgets.Layout(width = "825px", border='solid 2.0px white', margin='0px 10px 10px 10px', padding='5px 5px 5px 5px')
    
    
    
    
    text_dataset.observe(functools.partial( on_change_traj_data, query_index= query_index, cossine_matrix_1= Cossine_Matrix, cossine_matrix_2= CM, dtw_matrix= DTW_Matrix, edit_matrix= Edit_Matrix, 
                                           cossine_matrix_sensors_1= Cossine_Matrix_sensors, cossine_matrix_sensors_2= CM_sensors, euclidean_matrix_sensors= Eucli_Matrix_sensors, road_matrix_sensors= Road_Matrix_sensors,
                                           traj_data= traj_data, traj_data_sample= traj_data_sample, tokenizer_df= tokenizer_df, sensors_traj_sample= sensors_traj_sample, embedding_matrix= embedding_matrix,
                                           embedding= embedding, lat_long_1= LL, lat_long_2= LatLong, lat_long_sample= LatLong_sample, dataset_traj_number= Dataset, dataset_traj_number_geo25= Dataset_geo25, 
                                           traj_number= Traj_Number_Traj_id, sample_sensor_id_list= sts_id, object_loc= obj_loc, object_traj= obj_traj, objects_loc= Objects_loc, objects_traj= Objects_traj, 
                                           dict_traj_id= traj_id_dict, select_trajectory_dropdown= select_trajectory_dropdown, select_location_dropdown= select_location_dropdown, text_dataset= text_dataset, 
                                           embedding_dictionary= Emb_dict, out1= text_dataset_out, out2= select_trajectory_dropdown_output, out3= select_location_dropdown_output),names="value")

    
    
    select_trajectory_label = widgets.Label("Select Trajectory")
    select_trajectory_dropdown = widgets.Dropdown(description='',options=Objects_traj,layout=widgets.Layout(width="150px"))
    select_trajectory_dropdown.observe(functools.partial(on_change_object_traj, object_traj= obj_traj),names="value")
    select_trajectory_dropdown_output = out(select_trajectory_dropdown)
    
    select_location_label = widgets.Label("Select Loacation")
    select_location_dropdown = widgets.Dropdown(description='',options=Objects_loc,layout=widgets.Layout(width="150px"))
    select_location_dropdown.observe(functools.partial(on_change_object_loc, object_loc= obj_loc),names="value")
    select_location_dropdown_output = out(select_location_dropdown)

    top_k_trajectory_label = widgets.Label("Top-k")
    top_k_trajectory_text = widgets.Text(description='',layout=widgets.Layout(width="60px"))
    top_k_trajectory_text.observe(functools.partial(on_change_topk_traj, topk_loc= topk_loc),names="value")
    
    top_k_location_label = widgets.Label("Top-k")
    top_k_location_text = widgets.Text(description='',layout=widgets.Layout(width="60px"))
    top_k_location_text.observe(functools.partial(on_change_topk_loc, topk_traj= topk_traj),names="value")


    top_k_trajectory_button = widgets.Button(description="Plot", layout= widgets.Layout(width="60px"))
    top_k_trajectory_button.on_click(functools.partial(traj_topk_plot, dataset_traj_number= Dataset, object_traj= obj_traj, topk_traj= topk_traj, lat_long_sample= LatLong_sample,
                                  traj_number= Traj_Number_Traj_id, dict_traj_id= traj_id_dict, cossine_matrix= CM, out= out_topk_traj))
    top_k_trajectory_button.style.button_color = "lightgray"
    
    top_k_location_button = widgets.Button(description="Plot", layout= widgets.Layout(width="60px"))
    top_k_location_button.on_click(functools.partial(loc_topk_plot, cossine_matrix_sensors= CM_sensors, lat_long = LL, object_loc= obj_loc, topk_loc= topk_loc, sample_sensor_id_list= sts_id, tokenizer_df= tokenizer_df, out= out_topk_loc))
    top_k_location_button.style.button_color = "lightgray"
    
    mrr_trajectory_plot_button = widgets.Button(description="Plot", layout= widgets.Layout(width="60px"))
    mrr_trajectory_plot_button.on_click(functools.partial(traj_mrr_plot, cossine_matrix= Cossine_Matrix, edit_matrix= Edit_Matrix, dtw_matrix= DTW_Matrix, out= out_mrr_traj))
    mrr_trajectory_plot_button.style.button_color = "lightgray"
    
    mrr_location_plot_button = widgets.Button(description="Plot", layout=widgets.Layout(width="60px")) 
    mrr_location_plot_button.on_click(functools.partial(loc_mrr_plot, cossine_matrix= Cossine_Matrix_sensors, euclidean_matrix= Eucli_Matrix_sensors, road_matrix= Road_Matrix_sensors, out= out_mrr_loc))
    mrr_location_plot_button.style.button_color = "lightgray"
    
    mrr_trajectory_plot_box = widgets.VBox([spc, widgets.HBox([spc,spc,spc,widgets.HBox([widgets.Label("MRR"),spc,mrr_trajectory_plot_button])]),widgets.VBox([spc,spc,out_mrr_traj])])
    
    mrr_location_plot_box = widgets.VBox([spc, widgets.HBox([spc,spc,spc,widgets.HBox([widgets.Label("MRR"),spc,mrr_location_plot_button])]),widgets.VBox([spc,spc,out_mrr_loc])])
    
    similar_trajectory_box = widgets.HBox([widgets.VBox([widgets.HBox([widgets.HBox([select_trajectory_label, spc, select_trajectory_dropdown_output]),spc, widgets.HBox([top_k_trajectory_label,spc,top_k_trajectory_text,spc,spc,top_k_trajectory_button])]),
          widgets.HBox([spc,spc,spc,widgets.VBox([spc,out_topk_traj])])])])
    
    similar_location_box = widgets.VBox([widgets.VBox([widgets.HBox([widgets.HBox([select_location_label, spc, select_location_dropdown_output]),spc, widgets.HBox([top_k_location_label,spc,top_k_location_text,spc,spc,top_k_location_button])]),
          widgets.HBox([spc,spc,spc,widgets.VBox([spc,out_topk_loc])])])])
    
    accordion = widgets.Accordion(children=[mrr_location_plot_box,similar_location_box,mrr_trajectory_plot_box,similar_trajectory_box],selected_index=None)
    
    accordion.set_title(0, 'Mean Reciprocal Rank (MRR) Location')
    accordion.set_title(1, 'Similar Locations')
    accordion.set_title(2, 'Mean Reciprocal Rank (MRR) Trajectory')
    accordion.set_title(3, 'Similar Trajectories')
    

    intrinsic_evaluation_box = widgets.VBox([widgets.HBox([widgets.VBox([model_text,model_choice_out]),spc,spc,spc1, widgets.VBox([text_dataset_out])]),spc, accordion])
    
    intrinsic_evaluation_output.clear_output()
    
    display(intrinsic_evaluation_box)

def view_trajectory_data_output():

    out_df = out(widgets.Label(""))
    out_plot = out(widgets.Label(""))
    out_df.layout = widgets.Layout(border='solid 1.5px white',margin='0px 10px 10px 10px',padding='5px 5px 5px 5px')

    out_plot.layout = widgets.Layout(border='solid 2.0px white',margin='0px 10px 10px 10px',padding='5px 5px 5px 5px')
  
    out_statistics = widgets.Output()
    out_statistics.layout = widgets.Layout(border='solid 1.5px white',margin='10px 70px 10px 0px', padding='5px 5px 5px 5px')

    out_sampling = widgets.Output()
    out_sampling.layout = widgets.Layout(border='solid 1.5px white',margin='20px 10px 10px 0px', padding='5px 5px 5px 5px')

    sampling = 10

    with out_statistics:

        spc = widgets.Label("")
        spc1 = widgets.Label("",layout=widgets.Layout(width="1px",height="1px"))
        
        select_data = widgets.Label("Select Data")

        show_statistics_button = widgets.Button(description="Show Statistics", layout=widgets.Layout(width="120px"))
        show_statistics_button.on_click(functools.partial(show_statistics, dataset= dataset_2, out= out_df))
        show_statistics_button.style.button_color = "lightgray"

        aux = widgets.VBox(children=[widgets.HBox([spc,spc,select_data]),widgets.HBox(children=[spc, select_data_dropdown_out, spc,widgets.VBox([spc1,show_statistics_button])])])
        statistics_box = widgets.VBox(children=[aux,widgets.Label("",layout=widgets.Layout(width="19%")),out_df])

        display(statistics_box) 

    with out_sampling: 
        
        sampling_value = widgets.Label("Sampling")
        sampling_value_text  = widgets.Text(description='',layout=widgets.widgets.Layout(width="80px"))
        sampling_value_text.observe(functools.partial(on_change_sampling, sampling= sampling), names = 'value')

        sampling_button = widgets.Button(description="Plot Trajectories",layout=widgets.Layout(width="140px"))
        sampling_button.style.button_color = "lightgray"
        sampling_button.on_click(functools.partial(plot_sampling, sampling= sampling, dataset= dataset_2, out= out_plot))

        aux = widgets.VBox(children=[sampling_value,widgets.HBox(children=[sampling_value_text,sampling_button])])
        sampling_box = widgets.VBox(children=[aux,spc,out_plot])

        display(sampling_box)

    view_trajectory_data = widgets.HBox(children=[out_statistics,out_sampling])

    display(view_trajectory_data)

def extrinsic_evaluation_output():

    accuracy_output = widgets.Output()
    spc = widgets.Label(" ")
    spc1 = widgets.Label("",layout= widgets.Layout(width="102px"))
    
    model_data_df = pd.read_csv("Model#Data.csv")
    models_names = list(model_data_df.Model)
    
    extrinsic_evaluation_button = widgets.Button(description="Evaluation", layout=widgets.Layout(width="100px"))
    extrinsic_evaluation_button.style.button_color = "lightgray"
    extrinsic_evaluation_button.on_click( functools.partial(show_accuracy, select_mutiples_models= select_mutiples_models, select_mutiples_models_out= accuracy_output))

    display( widgets.VBox([widgets.HBox([select_mutiples_models_output,spc,spc,spc,extrinsic_evaluation_button]),spc, widgets.HBox([spc1, accuracy_output])]))