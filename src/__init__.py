import os
from src.data_pipeline import DefaulerData
from src.model_pipeline import DefaultModeler


def create_registry():
    for path in ["registry/data_pipeline", "registry/model_pipeline"]:
        dir  = ""
        for folder in path.split("/"):
            dir += f"{folder}/"
            if os.path.exists(dir) == False:
                os.mkdir(dir)
                
create_registry()