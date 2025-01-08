'''
File: imdb_api.py

Description: The primary API for interacting with the IMDb dataset.
'''

# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt

# Define API class
class IMDB_API:

    imdb = None  # dataframe

    def load_imdb(self, filename):
        '''
        Load in dataset as pandas dataframe
        '''
        self.imdb = pd.read_csv(filename)


    def get_columns(self):
        '''
        Get and return list of columns in dataset
        '''
        return list(self.imdb.columns)


    def prepare_data(self):
        '''
        Convert 'Gross', 'Released_Year', and 'Runtime' data into usable numeric data
        for analysis
        '''
        # Remove commas from 'Gross' column, convert to numeric data, and remove NaN values
        self.imdb['Gross'] = self.imdb['Gross'].replace(',', '', regex = True)
        self.imdb['Gross'] = pd.to_numeric(self.imdb['Gross'], errors = 'coerce')
        self.imdb.dropna(subset = ['Gross'], inplace = True)

        # Convert 'Released_Year' column to numeric data and remove NaN values. Convert column back to integers.
        self.imdb['Released_Year'] = pd.to_numeric(self.imdb['Released_Year'], errors = 'coerce')
        self.imdb.dropna(subset = ['Released_Year'], inplace = True)
        self.imdb['Released_Year'] = self.imdb['Released_Year'].astype(int)

        # Remove 'min' and whitespace from 'Runtime' column, convert to numeric data, and remove NaN values
        self.imdb['Runtime'] = self.imdb['Runtime'].str.replace(' min', '', regex = False)
        self.imdb['Runtime'] = pd.to_numeric(self.imdb['Runtime'], errors = 'coerce')
        self.imdb.dropna(subset = ['Runtime'], inplace = True)


    def get_unique_genres(self):
        '''
        Get and return list of unique genres from dataset
        '''
        # Split genres into lists and explode dataframe to get unique genres (documentation help from ChatGPT)
        unique_genres = self.imdb['Genre'].str.split(',').explode().str.strip().unique()

        return sorted(unique_genres.tolist())


    def filter_data(self, year_range, genre_selection, min_votes_slider):
        '''
        Filter data by year, genre, and vote selections and return filtered dataset
        '''
        # Filter data
        filtered_data = self.imdb[(self.imdb['Released_Year'] >= year_range[0]) &
                                 (self.imdb['Released_Year'] <= year_range[1])]
        if genre_selection:
            # Combine selected genres (separated by '|') and filter data if 'Genre' column contains those genre(s)
            # (documentation help from ChatGPT)
            pattern = '|'.join(genre_selection)
            filtered_data = filtered_data[filtered_data['Genre'].str.contains(pattern, na = False)]
        filtered_data = filtered_data[filtered_data['No_of_Votes'] >= min_votes_slider]

        return filtered_data


    def create_plot(self, plot_type, width, height, x_axis = None, y_axis = None, data = None, color = '#1f77b4',
                    edgecolor = 'None' ):
        '''
        Create and return a plot (scatterplot, barplot, or histogram) based on plot_type and
        parameter widget selections
        '''
        # Use original data if none is provided
        if data is None:
            data = self.imdb

        # Clear current figure
        plt.clf()

        # Create new figure
        plt.figure(figsize=(width / 100, height / 100))

        # Create plot_type
        if plot_type == 'Scatterplot':
            plt.scatter(data[x_axis], data[y_axis], alpha = 0.7, color = color, edgecolor = edgecolor)
            # Specify x-label for 'Runtime' to include units. For other columns, remove underscore from column name for
            # x-label and y-label. Use same method for title.
            if x_axis == 'Runtime':
                plt.xlabel('Runtime (mins)')
            else:
                plt.xlabel(x_axis.replace('_', ' '))
            plt.ylabel(y_axis.replace('_', ' '))
            plt.title(f'Scatter Plot of {y_axis.replace('_', ' ')} vs {x_axis.replace('_', ' ')}')

        elif plot_type == 'Barplot':
            data.groupby(x_axis)[y_axis].mean().plot(kind = 'bar', color = color, edgecolor = edgecolor)
            # Specify y-label for 'Gross' to include units. For other columns, remove underscore from column name
            # (x-labels) and add 'Average' (y-labels). Use same method for title.
            plt.xlabel(x_axis.replace('_', ' '))
            if y_axis == 'Gross':
                plt.ylabel('Average Gross (hundred millions)')
            else:
                plt.ylabel(f'Average {y_axis.replace('_', ' ')}')
            plt.title(f'Bar Plot of Average {y_axis.replace('_', ' ')} by {x_axis.replace('_', ' ')}')

        elif plot_type == 'Histogram':
            plt.hist(data[y_axis].dropna(), bins = 20, alpha = 0.7, color = color, edgecolor = edgecolor)
            # For x-label, remove underscore from y-axis column name. Specify y-label as 'Frequency' of y_axis data.
            # Use same method for title.
            plt.xlabel(y_axis.replace('_', ' '))
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {y_axis.replace('_', ' ')}')

        # Set x-tick rotation default value to horizontal
        plt.xticks(rotation = 0)

        return plt.gcf()


def main():

    # Create instance of IMDB_API
    imdb_api = IMDB_API()

    # Load IMDb dataset
    imdb_api.load_imdb('imdb_top_1000.csv')

    # Get and display columns in dataset
    columns = imdb_api.get_columns()
    print('Columns: ', columns, '\n')

    # Prepare data for use in dashboard creation
    imdb_api.prepare_data()

    # Get and display unique genres
    unique_genres = imdb_api.get_unique_genres()
    print('Unique genres: ', unique_genres, '\n')

    # Sample filter data and display filtered data
    year_range = (1990, 2020)
    genre_selection = ['Drama', 'Action', 'Fantasy']
    min_votes_slider = 100000
    filtered_data = imdb_api.filter_data(year_range, genre_selection, min_votes_slider)
    print('Filtered data: ', filtered_data, '\n')

    # Create sample plot
    if not filtered_data.empty:
        plot_type = 'Scatterplot'
        width = 600
        height = 600
        x_axis = 'Released_Year'
        y_axis = 'IMDB_Rating'
        sample_plot = imdb_api.create_plot(plot_type, width, height, x_axis, y_axis, filtered_data)
        plt.show()


if __name__ == '__main__':
    main()