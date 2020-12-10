import pandas as pd
import os
from data.data_deeplabcut import DataDLC
import numpy as np

class DatasetOpto:

    def __init__(self, cfg=None, db_path=None, data_path=None, project_path=None):
        if cfg is not None:
            db_path = cfg['db_path']
            data_path = cfg['data_path']
            project_path = cfg['project_path']
        self.db = pd.read_csv(db_path, sep=',')
        self.data_path = data_path
        self.project_path = project_path

    def get_all_sessions(self, stimhz=20):
        df_dict = {}
        stim_dict = {}
        offset_dict = {}
        references = self.get_sessions_by_reference()
        measures_folder = os.path.join(self.project_path, 'measures')

        for filename, row in references[references['Session'] == 1].iterrows():
            exp, sess, line = row['ExpNr'], row['Session'], row['MouseLine']
            if len(self.db[(self.db['fileName']==filename) &
                    (self.db['ExpNr']==exp) &
                    (self.db['Session']==sess) &
                    (self.db['MouseLine']==line) &
                    (self.db['Trial']==1) &
                    (self.db['StimHZ']==20)]) != 1:
                print(f"No valid session found for exp {exp}, sess {sess}, line {line}.")
                continue
            # if self.db row['StimHZ'] != 20:
            #     print(f"Session {line}: {exp}, {sess} does not start with a stimulation trial of 20Hz. Skipping ..")
            #     continue
            if line not in df_dict:
                df_dict[line] = {}
                stim_dict[line] = {}
                offset_dict[line] = {}
            filename = f"{row['MouseLine']}_{row['ExpNr']}_{row['Session']}.csv"
            filepath = os.path.join(measures_folder, filename)
            if not os.path.isfile(filepath):
                print(f"No measures file found for {filename}! Skipping ...")
                continue

            df = pd.read_csv(filepath, header=[0, 1], index_col=0)
            df, offset = self.get_offset_df(df)
            df_dict[line][(exp, sess)] = df
            stim_dict[line][(exp, sess)] = self.get_stim_times(line, exp, sess, stimhz=20, offset=offset)
            offset_dict[line][(exp, sess)] = offset
        return df_dict, stim_dict, offset_dict

    def get_offset_df(self, df, p_cutoff=0.9):
        """ Filter out low likelihood session start frames """
        p_cutoff = 0.9
        likelihoods = df.xs('likelihood', level=1, axis=1)
        offset = likelihoods[np.all(likelihoods >= p_cutoff, axis=1)].index[0]
        return df.copy().loc[offset:].reset_index(drop=True), offset

    def get_stim_times(self, line, exp, sess, stimhz=20, offset=0):
        trials = self.db[(self.db['MouseLine'] == line) & (self.db['ExpNr'] == exp) & (self.db['Session'] == sess) & (
                    self.db['StimHZ'] == stimhz)]
        if len(trials[trials['Trial']==1]) != 1:
            raise Exception(f"Problem with the trials for {line}: {exp}, {sess}. {len(trials[trials['Trial']==1])} trials found.")
        stim_times = list(zip(trials['StimStart']-offset, trials['StimEnd']-offset))
        return stim_times

    # def get_all_sessions(self):
    #     df_dict = {}
    #
    #     df_by_filename = self.db.groupby(['fileName', 'MouseLine']).first()
    #     for name, row in df_by_filename[df_by_filename['Session'] == 1].iterrows():
    #         exp, sess, line = row['ExpNr'], row['Session'], name[1]
    #         filename = name[0]
    #
    #         folder = self.find_behavior_folder(exp)
    #
    #         filename_measures = filename.split('.')[0] + '_measures.csv'
    #         filepath = os.path.join(folder, filename_measures)
    #         if not os.path.isfile(filepath):
    #             print(f"No measures file found for {filename_measures}! Skipping ...")
    #             continue
    #
    #         if line not in df_dict:
    #             df_dict[line] = {}
    #         df = pd.read_csv(filepath, header=[0, 1], index_col=0)
    #         df_dict[line][(exp, sess)] = df
    #     return df_dict

    def find_behavior_folder(self, exp):
        folder = os.path.join(self.data_path, str(int(exp)).zfill(5), 'behavior')
        if not os.path.isdir(folder):
            folder = os.path.join(self.data_path, str(int(exp)).zfill(5), 'Behavior')
            if not os.path.isdir(folder):
                folder = os.path.join(self.data_path, str(int(exp)).zfill(5))
        return folder

    def get_session_dlc(self, line, exp, sess, stimhz=20):
        trials = self.db[(self.db['MouseLine']==line) & (self.db['ExpNr'] == exp) & (self.db['Session'] == sess) & (self.db['StimHZ'] == stimhz)]
        filenames = trials['fileName'].unique()
        if len(filenames) > 1:
            raise Exception("Multiple filenames found for the given query.")
        elif len(filenames) == 0:
            raise Exception("No filenames found for the given query.")
        filename = filenames[0]
        folder = self.find_behavior_folder(exp)

        # Find DLC file
        files = [file for file in os.listdir(folder) if
                 file.endswith('.csv') and filename.split('.')[0] in file and 'DLC' in file]
        if len(files) > 1:
            raise Exception("Multiple DLC files found for", exp, sess, ":", files)
        elif len(files) == 0:
            print(f"No DLC file found for {filename}! Skipping ...")
        dlc_path = os.path.join(folder, files[0])
        data_dlc = DataDLC(dlc_path)
        stim_times = list(zip(trials['StimStart'], trials['StimEnd']))

        return data_dlc, stim_times

    def get_sessions_by_reference(self):
        return self.db.groupby(['fileName']).first()


    def get_video_path(self, line, exp, sess):
        trials = self.db[(self.db['MouseLine']==line) & (self.db['ExpNr']==exp) & (self.db['Session']==sess)]
        filenames = trials['fileName'].unique()
        if len(filenames) > 1:
            print(f"Multiple filenames found for the given query. ExpNr: {exp}, Session: {sess}. Continuing...")
        elif len(filenames) == 0:
            raise Exception("No filenames found for the given query.")
        filename = filenames[0]

        # Find folder
        folder = self.find_behavior_folder(exp)

        video_paths = [file for file in os.listdir(folder) if file.endswith('.avi') and filename.split('.')[0] in file][
            0]
        video_path = os.path.join(folder, video_paths)
        return video_path