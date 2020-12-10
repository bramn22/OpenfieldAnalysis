import os
import yaml
import numpy as np


def load_config(project_path=None):
    if project_path is None:
        config_path = "cfg.yaml"
    else:
        config_path = os.path.join(project_path, "cfg.yaml")
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def create_project_config(project_path):
    cfg = load_config()
    cfg['project_path'] = project_path
    with open(os.path.join(project_path, 'cfg.yaml'), 'w') as file:
        yaml.dump(cfg, file)


def split_data_segments(data, n=10):
    """ Segment the data (numpy matrix) with segments of size n. Returns for each row [Xi,Xi+1,..] with Xi being a row in the original matrix."""
    return np.reshape(data[:len(data)-(len(data)%n)], (-1, n*data.shape[1]), order='F')


