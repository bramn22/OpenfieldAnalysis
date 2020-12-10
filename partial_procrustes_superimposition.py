import pandas as pd
import numpy as np
import math
from tqdm import tqdm

def partial_procrustes_superimposition(df, fit_bodyparts, impose_bodyparts, ref):
    """ Execute partial procrustes superimposition on the given dataframe. The superimposition is done on all bodyparts,
     but only calculated on the bodyparts that are given in the 'fit_bodyparts' parameter. If None, it's all bodyparts"""
    df = df.copy() # Avoid overwriting original table
    df = _translate_to_centroid(df, fit_bodyparts, impose_bodyparts)
    df = _rotate_to_reference(df, ref, fit_bodyparts, impose_bodyparts)
    return df


def _translate_to_centroid(df, fit_bodyparts, impose_bodyparts):
    x_means = df[fit_bodyparts].xs('x', level=1, axis=1).mean(axis=1)
    y_means = df[fit_bodyparts].xs('y', level=1, axis=1).mean(axis=1)
    for bodypart in impose_bodyparts:
        df[(bodypart, 'x')] -= x_means
        df[(bodypart, 'y')] -= y_means
    df['T_x'] = x_means
    df['T_y'] = y_means

    return df


def _rotate_to_reference(df, ref, fit_bodyparts, impose_bodyparts):
    det = 0
    dot = 0
    for bodypart in fit_bodyparts:
        det += ((ref[bodypart, 'y'] * df[bodypart, 'x']) - (ref[bodypart, 'x'] * df[bodypart, 'y']))
        dot += ((ref[bodypart, 'x'] * df[bodypart, 'x']) + (ref[bodypart, 'y'] * df[bodypart, 'y']))
    angles = np.arctan2(det, dot)

    # EFFICIENT ROTATION
    rot_vector_x = np.array([np.cos(angles), -np.sin(angles)])
    rot_vector_y = np.array([np.sin(angles), np.cos(angles)])
    # df = df.reset_index()
    for bodypart in impose_bodyparts:
        new_vecs_x = df[bodypart][['x', 'y']].multiply(rot_vector_x.T, axis=1).sum(axis=1)
        new_vecs_y = df[bodypart][['x', 'y']].multiply(rot_vector_y.T, axis=1).sum(axis=1)
        df[bodypart, 'x'] = new_vecs_x
        df[bodypart, 'y'] = new_vecs_y
    # print(df)
    # df = df.set_index('index').droplevel(1, axis=1)
    # print(df)
    # SLOW ROTATION
    # rot_matrix = np.array([[np.cos(angles), -np.sin(angles)], [np.sin(angles), np.cos(angles)]])
    # # df.reset_index(drop=True) # TODO: check whether necessary!!
    # def rotate(row):
    #     row[['x', 'y']] = row[['x', 'y']] @ rot_matrix[:, :, row.name].T
    #     return row
    # df = df.reset_index()
    # for bodypart in impose_bodyparts:
    #     df[bodypart] = df[bodypart].apply(rotate, axis=1)
    # df = df.set_index('index')

    df['R'] = angles
    return df

def _rotate_to_xaxis(df, impose_bodyparts):
    body_vecs = df['nose'][['x', 'y']]-df['tail_start'][['x', 'y']]
    det = -body_vecs['y']
    dot = body_vecs['x']
    angles = np.arctan2(det, dot)

    # Efficient rotation
    rot_vector_x = np.array([np.cos(angles), -np.sin(angles)])
    rot_vector_y = np.array([np.sin(angles), np.cos(angles)])
    df = df.reset_index()
    for bodypart in impose_bodyparts:
        new_vecs_x = df[bodypart][['x', 'y']].multiply(rot_vector_x.T, axis=1).sum(axis=1)
        new_vecs_y = df[bodypart][['x', 'y']].multiply(rot_vector_y.T, axis=1).sum(axis=1)
        df[bodypart, 'x'] = new_vecs_x
        df[bodypart, 'y'] = new_vecs_y

    df = df.set_index('index')
    df['R_x'] = angles
    return df