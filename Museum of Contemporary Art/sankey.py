'''
File: sankey.py
Author: Sophie Sawyers

Description: A wrapper Library for plotly sankey visualizations and helper functions to prepare data for sankey
visualization
'''

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

pd.set_option('future.no_silent_downcasting', True)

# suppress SettingWithCopyWarning and FutureWarning warnings to avoid clutter (documentation help using ChatGPT)
pd.options.mode.chained_assignment = None
warnings.simplefilter(action = 'ignore', category = FutureWarning)

def aggregate_data(df, col1, col2, val_col):
    '''
    Aggregate data from df grouping by col1 and col2, adding a column val_col with size of groups.
    Return this data as a dataframe.
    '''
    # for gender data, convert 'male' values to 'Male' for proper aggregation
    if col1 or col2 == 'Gender':
        df.replace('male', 'Male', inplace = True)

    agg_data = df.groupby([col1, col2]).size().reset_index(name = val_col)

    return agg_data


def clean_data(df, count_threshold):
    '''
    Clean given dataframe to filter out rows where decade is 0 or there is missing/Nan data.
    Filter out rows where 'ArtistCount' is below count_threshold.
    Return resulting dataframe.
    '''
    if 'BirthDecade' in df.columns:
        df = df[df['BirthDecade'] != 0]

    # for nationality data, replace rows with values 'Nationality unknown' or 'Nationality Unknown' with NaN values
    df.replace(['Nationality unknown', 'Nationality Unknown'], np.nan, inplace = True)

    df = df.dropna()
    df = df[df['ArtistCount'] >= count_threshold]

    return df


def code_mapping(df, src, targ):
    '''
    Map labels in src and targ columns to integers.
    Return dataframe of src, targ, and vals columns as well as list of labels.
    '''
    # get distinct labels
    labels = sorted(set(df[src].astype(str).tolist() + df[targ].astype(str).tolist()))

    # create a label -> code mapping
    codes = range(len(labels))
    lc_map = dict(zip(labels, codes))

    # substitute codes for labels in dataframe (will map src and targ separately in case of different data types)
    df[src] = df[src].astype(str).map(lc_map)
    df[targ] = df[targ].astype(str).map(lc_map)

    return df, labels


def make_sankey(df, src, targ, *cols, vals = None, count_threshold = 20, **kwargs):
    '''
    Create and return a sankey figure using the following parameters:
    df - Dataframe
    src - Source node column
    targ - Target node column
    cols - Optional additional columns to be used for multi-layered sankey diagrams
    vals - Link values (ex: thickness, pad)
    count_threshold - Number of artists to filter data by for multi-layer diagrams; default is 20
    '''
    # if vals are provided and column name exists in dataframe, use corresponding column from dataframe. if not
    # provided, values will be length of dataframe.
    if vals and vals in df.columns:
        values = df[vals]
    else:
        values = [1] * len(df)

    # if cols are provided, go through stacking process
    if cols:
        # create empty dataframe for stacking and assign first src and targ to be paired (given src and targ)
        sankey_df = pd.DataFrame()
        current_src = src
        current_targ = targ

        # create first 'stack' for given src and targ. add result to empty dataframe. rename columns 'src' and 'targ'
        # for future mapping.
        initial_grouping = aggregate_data(df, current_src, current_targ, 'ArtistCount')
        initial_grouping = clean_data(initial_grouping, count_threshold)
        initial_grouping.rename(columns = {current_src: 'src', current_targ: 'targ'}, inplace = True)
        sankey_df = pd.concat([sankey_df, initial_grouping], ignore_index = True)

        for col in cols:
            # create next 'stack' by pairing given targ and new column (col). rename columns to 'src' and 'targ'
            # to match initial stack and add result to sankey_df, creating a stacked dataframe.
            col_grouping = aggregate_data(df, current_targ, col, 'ArtistCount')
            col_grouping = clean_data(col_grouping, count_threshold)
            col_grouping.rename(columns = {current_targ: 'src', col: 'targ'}, inplace = True)
            sankey_df = pd.concat([sankey_df, col_grouping], ignore_index = True)

            # shift columns so new source becomes old target and new target is next column in col
            current_targ = col
    else:
        # keep sankey_df as given df if no cols are given. rename columns to 'src' and 'targ' for mapping.
        sankey_df = df.copy()
        sankey_df.rename(columns = {src: 'src', targ: 'targ'}, inplace = True)

    # re-adjust 'values' values after stacking
    if vals and vals in sankey_df.columns:
        values = sankey_df[vals]
    else:
        values = [1] * len(sankey_df)

    # map 'src' and 'targ' columns to integers
    sankey_df, labels = code_mapping(sankey_df, 'src', 'targ')

    # initial link using 'src', 'targ', and values columns
    link = {'source': sankey_df['src'], 'target': sankey_df['targ'], 'value': values}

    # modify sankey using kwargs (if provided). otherwise, set to default values (larger thickness for visibility).
    thickness = kwargs.get('thickness', 70)
    pad = kwargs.get('pad', 50)

    node = {'label': labels, 'thickness': thickness, 'pad': pad}

    sk = go.Sankey(link = link, node = node)
    fig = go.Figure(sk)

    # set figure to display separately in browser
    fig.show(renderer = 'browser')
