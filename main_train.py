import pandas as pd
import numpy as np

""" Currently only works for Anna's file naming """
data_path = r'Y:\Data'
exp_list = [('01317', '01'), ('01314', '01')]

""" Load data ()"""
videos_dict = {}
dlcs_dict = {}
for exp, sess in exp_list:
    videos_dict[(exp, sess)] = dataset.load_video(data_path, exp, sess)
    dlcs_dict[(exp, sess)] = dataset.load_dlc(data_path, exp, sess)
    dlcs_dict[(exp, sess)].df['led'] = dataset.load_led(data_path, exp, sess)

bodyparts = dlcs_dict[exp_list[0]].columns.levels[0].drop('bodyparts')
bodyparts = bodyparts.drop('led')
bodyparts_body = bodyparts.drop('tail_end')

# Combine all dataframes
df_all = pd.concat([dlc.df for dlc in list(dlcs_dict.values())])
# Remove poses with low likelihood bodyparts
# p = 0.9
# df_all = df_all[np.all(df_all.xs('likelihood', level=1, axis=1)>0.9, axis=1)]

""" Correct outliers """
p = 0.9
for exp, sess in exp_list:
    dlcs_dict[(exp, sess)].filter_likelihood(p)

""" Choose reference pose """
idxs = [500, 1000, 1500, 5000]
fig, axs = plt.subplots(2, 2, figsize=(5, 5), sharex=True, sharey=True)
for i, idx in enumerate(idxs):
    axs[i//2][i%2].scatter(df_all.iloc[idx].xs('x', level=1, axis=0), df_all.iloc[idx].xs('y', level=1, axis=0))
plt.show()


""" Only for training data: """
# Load data
# Correct outliers with filtering
# Choose reference frame
# Run PPS
# Calculate SSM


""" For each session: """
# Load data
# Correct outliers with filtering
# Run PPS
# Correct outliers using SSM
# Calculate measures using df and SSM
# Normalize measures based on all data/subset
