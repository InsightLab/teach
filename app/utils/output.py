from IPython.display import display
import ipywidgets as widgets


def out(ds):
    
    out = widgets.Output()
    with out:
        display(ds)
    return out