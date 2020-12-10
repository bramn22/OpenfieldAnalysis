import os
from utils import utils


def create_project(project_path):
    if os.path.isdir(project_path):
        raise Exception("Project already exists.")
    analysis_path = os.path.join(project_path, 'analysis')
    analysis_full_path = os.path.join(analysis_path, 'full')
    analysis_subsec_path = os.path.join(analysis_path, 'subsec')
    analysis_full_clustering_path = os.path.join(analysis_full_path, 'clustering')
    analysis_subsec_clustering_path = os.path.join(analysis_subsec_path, 'clustering')
    boxes_path = os.path.join(project_path, 'boxes')
    measures_path = os.path.join(project_path, 'measures')
    preprocessed_path = os.path.join(project_path, 'preprocessed')
    os.mkdir(project_path)
    os.mkdir(boxes_path)
    os.mkdir(measures_path)
    os.mkdir(preprocessed_path)
    os.mkdir(analysis_path)
    os.mkdir(analysis_full_path)
    os.mkdir(analysis_subsec_path)
    os.mkdir(analysis_full_clustering_path)
    os.mkdir(analysis_subsec_clustering_path)

    # Add config file
    utils.create_project_config(project_path)
    return project_path


def load_project(project_path):
    return utils.load_config(project_path)
