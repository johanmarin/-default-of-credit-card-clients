import os
import joblib

def registry_object(obj, registry_path):
    """
    Save an object using joblib.dump to a given file path.

    Args:
        obj (Any): The object to be saved.
        registry_path (str): The file path where the object will be saved.

    Returns:
        None
    """
    if registry_path:
        joblib.dump(obj, registry_path)
        
def get_registred_object(registry_path):
    """
    Load an object previously saved using joblib.dump from a given file path.

    Args:
        registry_path (str): The file path where the object was saved.

    Returns:
        The object previously saved.
    """
    if os.path.exists(registry_path):
        return joblib.load(registry_path)
    else:
        raise ValueError(f"file {registry_path} not exists, you need train the model before using this") 
    