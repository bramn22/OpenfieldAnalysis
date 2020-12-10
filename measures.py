import numpy as np
import pandas as pd


# def post_pc1(df):
#     pass

# def post_pc2(df):
#     pass

# def mov_pc1(df):
#     pass

# def mov_pc2(df):
#     pass

def post_elongation(df):
    """ Euclidean distance between nose and tail base (todo: improve) """
    df['measures', 'post_elongation'] = np.linalg.norm(df['nose'][['x', 'y']] - df['tail_start'][['x', 'y']], axis=1)
    return df


def post_bend(df):
    """ Angle of <tail base, neck, nose> """
    neck = (df['ear_left'][['x', 'y']] + df['ear_right'][['x', 'y']]) / 2
    nose = df['nose'][['x', 'y']] - neck
    tail_start = df['tail_start'][['x', 'y']] - neck

    print('nose', nose.iloc[0])
    print('tail_start', tail_start.iloc[0])
    # Dot product:
    dot = np.sum(nose.multiply(tail_start), axis=1)
    print('dot', dot.iloc[0])

    print('magn', (np.linalg.norm(nose, axis=1) * np.linalg.norm(tail_start, axis=1))[0])

    df['measures', 'post_bend'] = np.pi - np.arccos(dot / (np.linalg.norm(nose, axis=1) * np.linalg.norm(tail_start, axis=1)))
    return df


def post_bend_dir(df):
    """ Angle of <tail base, neck, nose> - directionality"""
    neck = (df['ear_left'][['x', 'y']] + df['ear_right'][['x', 'y']]) / 2
    nose = df['nose'][['x', 'y']] - neck
    tail_start = df['tail_start'][['x', 'y']] - neck

    print('nose', nose.iloc[0])
    print('tail_start', tail_start.iloc[0])
    dot = np.sum(nose.multiply(tail_start), axis=1)
    det = nose['x'] * tail_start['y'] - nose['y'] * tail_start['x']
    angle = np.arctan2(det, dot)

    df['measures', 'post_bend_D'] = np.sign(angle) * (np.pi - np.abs(angle))
    return df


def mov_locomotion(df):
    """ Euclidian distance between translation points (centroids) of subsequent frames """
    diff = df[['T_x', 'T_y']].diff()
    df['measures', 'mov_locomotion'] = np.linalg.norm(diff, axis=1)
    return df


def mov_acceleration(df):
    diff = df['measures', 'mov_locomotion'].diff()
    df['measures', 'mov_acceleration'] = diff
    return df


def mov_freeze(df, bodyparts):
    """ Euclidian distance between all bodyparts of subsequent frames """
    diff_x = df[bodyparts].xs('x', level=1, axis=1).diff()
    diff_y = df[bodyparts].xs('y', level=1, axis=1).diff()
    diff = pd.concat((diff_x, diff_y), axis=1)
    df['measures', 'mov_freeze'] = -np.linalg.norm(diff, axis=1)
    return df


def mov_rotation(df):
    """ Absolute difference between subsequent rotation angles (TODO: use rotation matrix like in the paper?) """
    diff = df['R'].diff()
    df['measures', 'mov_rotation'] = np.abs(diff)
    return df


def mov_rotation_dir(df):
    """ Absolute difference between subsequent rotation angles (TODO: use rotation matrix like in the paper?) """
    diff = df['R'].diff()
    df['measures', 'mov_rotation_D'] = diff
    return df


def mov_elongation(df):
    diff = df['measures', 'post_elongation'].diff()
    df['measures', 'mov_elongation'] = diff
    return df


def mov_bend(df):
    diff = df['measures', 'post_bend'].diff()
    df['measures', 'mov_bend'] = np.abs(diff)
    return df


def mov_bend_dir(df):
    diff = df['measures', 'post_bend'].diff()
    df['measures', 'mov_bend_D'] = diff
    return df

def distance_from_corner(df, corners):
    distances = []
    for corner in corners:
        corner_tiled = np.tile(corner, (len(df), 1))
        distance = np.linalg.norm(df[['T_x', 'T_y']] - corner_tiled, axis=1)
        distances.append(distance)
    min_distances = np.min(np.vstack(distances), axis=0)
    print(df.shape)
    print(min_distances.shape)
        # np.min([corner for corner in corners])
    df['measures', 'corner_distance'] = min_distances
    return df

def mov_locomotion_side(df):
    orient_vectors = df['nose'][['x', 'y']] - df['tail_start'][['x', 'y']]

    angles = -np.squeeze(df['R'])

    rot_vector_x = np.array([np.cos(angles), -np.sin(angles)])
    print(rot_vector_x.shape)
    rot_vector_y = np.array([np.sin(angles), np.cos(angles)])
    orient_vectors_R = orient_vectors.copy().reset_index()
    orient_vectors_R_temp = orient_vectors_R[['x', 'y']]
    new_vecs_x = orient_vectors_R_temp.multiply(rot_vector_x.T, axis=1).sum(axis=1)
    new_vecs_y = orient_vectors_R_temp.multiply(rot_vector_y.T, axis=1).sum(axis=1)
    orient_vectors_R['x'] = new_vecs_x
    orient_vectors_R['y'] = new_vecs_y

    # rot_matrix = np.array([[np.cos(angles), -np.sin(angles)], [np.sin(angles), np.cos(angles)]])

    # def rotate(row):
    #     row[['x', 'y']] = row[['x', 'y']] @ rot_matrix[:, :, row.name].T
    #     return row
    # orient_vectors_R = orient_vectors.copy().reset_index()
    # orient_vectors_R = orient_vectors_R.apply(rotate, axis=1)

    # Shift all values up, because we want the orientation at the previous frame
    # print(orient_vectors_R)
    orient_vectors_R = orient_vectors_R.set_index('index') # Remove?
    orient_vectors_R = pd.concat([pd.DataFrame({'x': [np.nan], 'y': [np.nan]}), orient_vectors_R.iloc[:-1]], ignore_index=True)
    transl_vectors = df[['T_x', 'T_y']].diff().reset_index(drop=True)
    transl_vectors = transl_vectors.droplevel(level=1, axis=1)

    dot = orient_vectors_R['x'] * transl_vectors['T_x'] + orient_vectors_R['y'] * transl_vectors['T_y']
    det = orient_vectors_R['x'] * transl_vectors['T_y'] - orient_vectors_R['y'] * transl_vectors['T_x']
    angle = np.arctan2(det, dot)

    transl_vectors = transl_vectors.rename(columns={"T_x": "x", "T_y": "y"})
    orth_proj = orient_vectors_R.multiply(
        np.sum(orient_vectors_R.multiply(transl_vectors), axis=1) / np.sum(orient_vectors_R.multiply(orient_vectors_R),
                                                                           axis=1), axis=0)
    lateral_distance = np.linalg.norm(transl_vectors - orth_proj, axis=1)
    lateral_distance = lateral_distance * np.sign(angle) * -1
    df['measures', 'mov_locomotion_side_angle'] = angle
    df['measures', 'mov_locomotion_side_dist'] = lateral_distance
    return df


def calc_measures(df, bodyparts, corners=None):
    df = df.copy()  # Avoid overwriting original table
    df = post_elongation(df)
    df = post_bend(df)
    df = post_bend_dir(df)
    df = mov_locomotion(df)
    df = mov_acceleration(df)
    df = mov_freeze(df, bodyparts)
    df = mov_rotation(df)
    df = mov_rotation_dir(df)
    df = mov_elongation(df)
    df = mov_bend(df)
    df = mov_bend_dir(df)
    df = mov_locomotion_side(df)
    if corners is not None:
        df = distance_from_corner(df, corners)
    return df


# ----------------- Normalize ------------------------------
def normalize_zscore(df, means=None, stds=None):
    df = df.copy()
    if means is None or stds is None:
        means = df['measures'].mean(axis=0)
        stds = df['measures'].std(axis=0)
    print('means', means)
    print('stds', stds)
    print(means.shape)
    df['measures'] = (df['measures'] - means).divide(stds)
    return df, means, stds


def normalize_quantiles(df, n_intervals=101):
    intervals = np.linspace(0, 1, num=n_intervals)
    values = np.squeeze(np.repeat(intervals[:, None], np.ceil(len(df) / n_intervals), axis=0))[:len(df)]
    df = df.copy()
    for col in df['measures']:
        print(col)
        df = df.sort_values(by=[('measures', col)], axis=0)
        df[('measures', col)] = values
    return df.sort_index()


def normalize_scaling(df, n=100):
    df = df.copy()
    for col in df['measures']:
        print(col)
        df = df.sort_values(by=[('measures', col)], axis=0)
        print(df[('measures', col)].iloc[:n])
        min_v = df[('measures', col)].iloc[:n].mean()
        max_v = df[('measures', col)].iloc[-n:].mean()
        print('min_v', min_v)
        print('max_v', max_v)
        df[('measures', col)] = np.clip((df[('measures', col)] - min_v) / abs(max_v), 0, 1)
    return df.sort_index()
