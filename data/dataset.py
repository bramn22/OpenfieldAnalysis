import os
from data import data_video, data_deeplabcut
import pandas as pd

def load_video(data_path, exp_id, sess_id):
    folder = os.path.join(data_path, exp_id, 'behavior')
    files = [file for file in os.listdir(folder) if file.endswith('.avi') and '_'+sess_id+'_' in file]
    if len(files) > 1:
        raise Exception("Multiple video files found for", exp_id, sess_id, ":", files)
    filepath = os.path.join(folder, files[0])
    return data_video.DataVideo(filepath)

def load_dlc(data_path, exp_id, sess_id):
    folder = os.path.join(data_path, exp_id, 'behavior')
    files = [file for file in os.listdir(folder) if file.endswith('.csv') and '_'+sess_id+'_' in file and 'DLC' in file]
    if len(files) > 1:
        raise Exception("Multiple DLC files found for", exp_id, sess_id, ":", files)
    filepath = os.path.join(folder, files[0])
    return data_deeplabcut.DataDLC(filepath)

def load_led(data_path, exp_id, sess_id):
    folder = os.path.join(data_path, exp_id, 'behavior')
    files = [file for file in os.listdir(folder) if file.endswith('.csv') and '_'+sess_id+'_' in file and 'LED' in file]
    if len(files) > 1:
        raise Exception("Multiple LED files found for", exp_id, sess_id, ":", files)
    filepath = os.path.join(folder, files[0])
    return pd.read_csv(filepath, names=['LED'])
