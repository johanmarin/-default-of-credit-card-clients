import joblib

def registry_object(obj, registry_path):
    if registry_path:
        joblib.dump(obj, registry_path)
        
def get_registred_object(registry_path):
    return joblib.load(registry_path)