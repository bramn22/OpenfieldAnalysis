import partial_procrustes_superimposition as pps
import SSM
import measures
import os
import pandas as pd
from utils import box_detection

def preprocess_df(df, fit_bodyparts, impose_bodyparts, ref_dlc):
    # PPS
    df_pps = pps.partial_procrustes_superimposition(df, fit_bodyparts, impose_bodyparts, ref=ref_dlc)
    # SSM
    # df_pps_ssm = ssm.transform(df_pps, bodyparts_body)
    # Measures
    # df_final = measures.calc_measures(df_pps_ssm, bodyparts)
    return df_pps

def preprocess_all(cfg, dataset, overwrite=False, detect_boxes=True):
    # Optional: Create preprocessed folder in the project
    project_path = cfg['project_path']
    preproc_path = os.path.join(project_path, 'preprocessed')
    ref_dlc = pd.read_csv(os.path.join(project_path, 'ref_pose.csv'), index_col=0, header=[0, 1]).iloc[0]

    if not os.path.isdir(preproc_path):
        print("No previous preprocessed data found. Creating preprocessed folder in Project.")
        os.mkdir(preproc_path)
    else:
        print("Previously preprocessed data found in Project.")

    # Preprocess all sessions
    references = dataset.get_sessions_by_reference()
    for filename, row in references[references['Session'] == 1].iterrows():
        filename = f"{row['MouseLine']}_{row['ExpNr']}_{row['Session']}.csv"
        if not overwrite and os.path.isfile(os.path.join(preproc_path, filename)):
            print(f"Already processed: {filename}. To overwrite, set the overwrite parameter to True.")
            continue
        print(f"Loading file {filename} ...")
        try:
            data_dlc, stim_times = dataset.get_session_dlc(row['MouseLine'], row['ExpNr'], row['Session'], stimhz=20)
            data_dlc.filter_likelihood(p=0.9)
            # Preprocess file
            df = preprocess_df(data_dlc.df, cfg['fit_bodyparts'], cfg['impose_bodyparts'], ref_dlc)
            # Save preprocessed file
            new_filepath = os.path.join(preproc_path, filename)
            df.to_csv(new_filepath)
        except Exception:
            print(f"Could not load {filename}! Check whether it has a trial with 20Hz.")
    if detect_boxes:
        boxes_folder = os.path.join(project_path, 'boxes')

        if not os.path.isdir(boxes_folder):
            os.mkdir(boxes_folder)

        box_df = pd.DataFrame(columns=['MouseLine', 'ExpNr', 'Session', 'Corners'])
        for i, (filename, row) in enumerate(references[references['Session'] == 1].iterrows()):
            video_path = dataset.get_video_path(row['MouseLine'], row['ExpNr'], row['Session'])
            try:
                corners = box_detection.detect(video_path, save_path=os.path.join(boxes_folder, f"{row['MouseLine']}_{row['ExpNr']}_{row['Session']}.png"))
                print(corners)
                box_df.loc[i] = [row['MouseLine'], row['ExpNr'], row['Session'], str(corners.tolist())]
            except Exception as e:
                print(e, filename)
        box_df.to_csv(os.path.join(boxes_folder, 'boxes.csv'))

def calculate_measures(cfg, dataset):
    import ast

    # Calculate measures for all data in the preprocessed folder
    project_path = cfg['project_path']
    measures_path = os.path.join(project_path, 'measures')
    preproc_path = os.path.join(project_path, 'preprocessed')

    if not os.path.isdir(measures_path):
        print("No previous measures data found. Creating measures folder in Project.")
        os.mkdir(measures_path)
    else:
        print("Previously measures data found in Project.")

    boxes_csv = pd.read_csv(os.path.join(project_path, 'boxes', 'boxes.csv'), index_col=0)
    references = dataset.get_sessions_by_reference()
    for filename, row in references[references['Session'] == 1].iterrows():
        preproc_filename = f"{row['MouseLine']}_{row['ExpNr']}_{row['Session']}.csv"
        if os.path.isfile(os.path.join(preproc_path, preproc_filename)):
            print(f"Loading file {preproc_filename} ...")
            df = pd.read_csv(os.path.join(preproc_path, preproc_filename), header=[0, 1], index_col=0)
            file_corners = boxes_csv[(boxes_csv['MouseLine']==row['MouseLine']) & (boxes_csv['ExpNr']==row['ExpNr']) & (boxes_csv['Session']==row['Session'])]
            corners = ast.literal_eval(file_corners['Corners'].iloc[0])
            df = measures.calc_measures(df, cfg['impose_bodyparts'], corners)
            new_filepath = os.path.join(measures_path, preproc_filename)
            df.to_csv(new_filepath)
        else:
            print(f"No preprocessed file found for {preproc_filename}")


    preprocessed_files = os.listdir(preproc_path)
    for file in preprocessed_files:
        df = pd.read_csv(os.path.join(preproc_path, file), header=[0, 1], index_col=0)
        df = measures.calc_measures(df, cfg['impose_bodyparts'], corners)
        new_filepath = os.path.join(measures_path, file)
        df.to_csv(new_filepath)
