
import pandas as pd
import sankey as sk

def main():

    # Step 1:
    # load the data from the json file
    artist_df = pd.read_json('artists.json')

    # convert data into pandas dataframe containing three columns
    final_artist_df = artist_df[['Nationality', 'Gender', 'BeginDate']]

    # convert birth year into birth decade and drop birth year column
    final_artist_df['BirthDecade'] = (final_artist_df['BeginDate'] // 10) * 10
    final_artist_df.drop(columns = ['BeginDate'], inplace = True)

    # Steps 2-5:
    # aggregate data by nationality and decade
    nationality_decade_agg = sk.aggregate_data(final_artist_df, 'Nationality', 'BirthDecade',
                                               'ArtistCount')

    # filter data for missing/invalid values and apply artist count threshold
    nationality_decade_agg = sk.clean_data(nationality_decade_agg, 20)

    # generate a sankey diagram with nationality as sources and birth decade as targets
    sk.make_sankey(nationality_decade_agg, 'Nationality', 'BirthDecade', vals = 'ArtistCount')

    # Step 6:
    # aggregate data by nationality and gender
    nationality_gender_agg = sk.aggregate_data(final_artist_df, 'Nationality', 'Gender', 'ArtistCount')

    # filter data for missing/invalid values and apply artist count threshold (used higher threshold for visibility)
    nationality_gender_agg = sk.clean_data(nationality_gender_agg, 30)

    # generate a sankey diagram with nationality as sources and gender as targets
    sk.make_sankey(nationality_gender_agg, 'Nationality', 'Gender', vals = 'ArtistCount')

    # Step 7:
    # aggregate data by gender and decade
    gender_decade_agg = sk.aggregate_data(final_artist_df, 'Gender', 'BirthDecade', 'ArtistCount')

    # filter data for missing/invalid values and apply artist count threshold
    gender_decade_agg = sk.clean_data(gender_decade_agg, 20)

    # generate a sankey diagram with gender as sources and birth decade as targets
    sk.make_sankey(gender_decade_agg, 'Gender', 'BirthDecade', vals = 'ArtistCount')

    # Step 8:
    # generate multi-layered sankey diagram using three layers: nationality, gender, and birth decade. use nationality
    # as the source, gender as the target, and birth decade as the additional cols parameter (used higher threshold for
    # visibility).
    final_artist_df['Gender'].replace('male', 'Male', inplace = True)
    sk.make_sankey(final_artist_df, 'Nationality', 'Gender', 'BirthDecade', vals = 'ArtistCount',
                   count_threshold = 30)

main()
