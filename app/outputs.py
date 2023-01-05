from IPython.display import display, HTML
import ipywidgets as widgets

def out(df):
    
    out = widgets.Output()
    with out:
        display(df)
    return out