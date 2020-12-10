from data import dataset_opto
from videoplayer import trial_viewer
import utils.box_detection as box_detection
import project
import analysis
import os

data_path = r'Y:\Data'
db_path = r'Y:\Data\Ania\opto-fUSi\theOnlyTrueDB.csv'

# Create project
project_path = r'E:\dlc\OpenfieldOpto'
# project_path = project.create_project(project_path)
cfg = project.load_project(project_path)
dataset = dataset_opto.DatasetOpto(cfg=cfg)


# analysis.preprocess_all(cfg, dataset)

# Check trial by trial videos
exp, sess, line = 863, 1, 'PV'
data_dlc, stim_times = dataset.get_session_dlc(line, exp, sess, 20)
video_path = dataset.get_video_path(line, exp, sess)
trial_viewer.play_DLC_video_trials(dlc=data_dlc, video_path=video_path, trial_times=stim_times)

# corners = box_detection.detect(video_path)
# print(corners)