'''
File: imdb_explorer.py

Description: The main application for creating interactive dashboard for IMDb dataset.
'''

# Import necessary packages
import panel as pn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imdb_api import IMDB_API

# Loads javascript dependencies and configures Panel (required)
pn.extension()

# INITIALIZE API
api = IMDB_API()
api.load_imdb('imdb_top_1000.csv')
api.prepare_data()


# WIDGET DECLARATIONS

# Search widgets:

# Implement widget allowing user to select desired plot type
plot_type = pn.widgets.RadioBoxGroup(name = 'Plot type: ', options = ['Scatterplot', 'Barplot', 'Histogram'],
                                     value = 'Scatterplot', inline = False, margin = (10, 0, 0, 10))

# Separate x and y selectors for each plot type, set default values to reasonable quantitative columns from dataset
x_axis_selection = pn.widgets.Select(name = 'X-axis: ', options = [], value = api.get_columns()[2],
                                     margin = (0, 0, 5, 10))
y_axis_selection = pn.widgets.Select(name = 'Y-axis: ', options = [], value = api.get_columns()[6],
                                     margin = (0, 0, 10, 10))

# Implement widgets to restrict plotting data based on respective column max and min values or all column values (genre)
year_range = pn.widgets.IntRangeSlider(name = 'Year range: ', start = 1920, end = 2020, value = (1920, 2020), step = 1)
min_votes_slider = pn.widgets.IntSlider(name = 'Minimum votes: ', start = 25000, end = 2343200, value = 0, step = 1)
genre_selection = pn.widgets.MultiSelect(name = 'Genre(s): ', options = api.get_unique_genres(), value = [],
                                         margin = (5, 0, 10, 10))

# Plotting widgets:

# Implement width and height widgets with values based on display size
width = pn.widgets.IntSlider(name = 'Width: ', start = 500, end = 1000, value = 750, step = 50, margin = (10, 0, 0, 10))
height = pn.widgets.IntSlider(name = 'Height: ', start = 200, end = 1200, value = 400, step = 50)

# Implement widgets for visual plot customization and plot legibility (x-tick tilt/size/display/skip)
# Note: default color_picker value is matplotlib default blue hex code, #1f77b4
color_picker = pn.widgets.ColorPicker(name = 'Color: ', value = '#1f77b4')
border_checkbox = pn.widgets.Checkbox(name = 'Add border?', value = False, margin = (15, 0, 5, 10))

tilt_x_ticks = pn.widgets.Checkbox(name = 'Tilt x-axis ticks?', value = True, margin = (10, 0, 5, 10))
x_tick_font_size = pn.widgets.IntSlider(name = 'X-axis tick font size: ', start = 4, end = 20, value = 8, step = 1,
                                        margin = (10, 0, 10, 10))
show_all_x_ticks = pn.widgets.Checkbox(name = 'Show all x-axis ticks?', value = False, margin = (10, 0, 5, 10))
tick_skip_slider = pn.widgets.IntInput(name = 'Skip every n x-axis ticks: ', start = 2, end = 20, value = 4,
                                       margin = (10, 0, 10, 10))

# Table widgets:

# Implement widgets for customizing data displayed in table (series and overview)
include_series_checkbox = pn.widgets.Checkbox(name = 'Include series title?', value = False, margin = (10, 0, 0, 10))
include_overview_checkbox = pn.widgets.Checkbox(name = 'Include overview?', value = False, margin = (10, 0, 5, 10))


# CALLBACK FUNCTIONS

def update_axis_options(plot_type):
    '''
    Update x- and y-axis 'options' based on type of plot selected
    '''
    x_options = []
    y_options = []

    # Update x_options and y_options based on plot type
    if plot_type == 'Scatterplot':
        x_options = ['Released_Year', 'Runtime']
        y_options = ['IMDB_Rating', 'Meta_score']
    elif plot_type == 'Barplot':
        x_options = ['Genre', 'Released_Year']
        y_options = ['IMDB_Rating', 'Gross']
    elif plot_type == 'Histogram':
        # Do not need x_options for this plot (leave as empty list)
        y_options = ['IMDB_Rating', 'Runtime', 'Meta_score']

    # Update widget 'options'
    x_axis_selection.options = x_options
    y_axis_selection.options = y_options

    # Create helper function to update x-axis and y-axis options based on selected plot type
    def update_selection(current_value, options_list, axis_selection):
        '''
        Set axis selection 'value' to current 'value' if 'value' is in 'options' list created above.
        If not in list, set 'value' to first item in axis 'options' list.
        '''
        if options_list:
            axis_selection.value = current_value if current_value in options_list else options_list[0]

    # Update axis options
    update_selection(x_axis_selection.value, x_options, x_axis_selection)
    update_selection(y_axis_selection.value, y_options, y_axis_selection)


def generate_table(x_axis_selection, y_axis_selection, year_range, min_votes_slider, genre_selection,
                   include_series_checkbox, include_overview_checkbox):
    '''
    Generate and return datatable in 'Table' tab of dashboard based on selections
    '''
    # Filter data by year, genre, and vote conditions
    filtered_data = api.filter_data(year_range, genre_selection, min_votes_slider)

    # If filtered data is empty (no data meets all conditions), return error message
    if filtered_data.empty:
        return pn.pane.Markdown('### No data found matching the selected criteria.')

    # Ensure y_axis_selection is not None (needed in all plot types)
    if y_axis_selection is None:
        print('Y selection is None, cannot generate datatable.')

    # Initialize columns to include in datatable
    columns = [y_axis_selection]

    # Handle histogram plots where x_axis_selection is None by calculating y_axis_selection frequency for table
    if x_axis_selection is None:
        frequency_data = filtered_data[y_axis_selection].value_counts().reset_index()
        frequency_data.columns = [y_axis_selection, 'Frequency']
        local = frequency_data.dropna()

        # Include title and/or overview if checkbox is checked by merging local dataframe with filtered_data
        additional_columns = []
        if include_series_checkbox:
            additional_columns.append('Series_Title')
        if include_overview_checkbox:
            additional_columns.append('Overview')

        if additional_columns:
            # documentation help from ChatGPT
            merge_columns = [y_axis_selection] + additional_columns
            local = pd.merge(local, filtered_data[merge_columns].dropna(), on = y_axis_selection, how = 'left')
    else:
        # If x_axis_selection or y_axis_selection are not valid, print error message
        if x_axis_selection not in api.imdb.columns or y_axis_selection not in api.imdb.columns:
            print(f'Selected values are not in dataframe columns: {x_axis_selection}, {y_axis_selection}')
        # If x_axis_selection and y_axis_selection are valid, add x_axis_selection to columns with y_axis_selection
        columns.insert(0, x_axis_selection)
        local = filtered_data[columns].dropna()

        # Include title and/or overview if checkbox is checked
        if include_series_checkbox or include_overview_checkbox:
            additional_columns = []
            if include_series_checkbox:
                additional_columns.append('Series_Title')
            if include_overview_checkbox:
                additional_columns.append('Overview')

            if additional_columns:
                local = pd.concat([local, filtered_data[additional_columns]], axis = 1)

    # Create and return datatable
    table = pn.widgets.Tabulator(local, selectable = False, show_index = False, pagination = None)
    return table


def generate_plot(plot_type, x_axis_selection, y_axis_selection, year_range, min_votes_slider, genre_selection, width,
                  height, color_picker, border_checkbox, tilt_x_ticks, x_tick_font_size, show_all_x_ticks,
                  tick_skip_slider):
    '''
    Generate and return plot in 'Plot' tab of dashboard based on selections
    '''
    # Clear current plot and update axis options
    plt.clf()
    update_axis_options(plot_type)

    # Ensure x_axis_selection is valid if appropriate. If not valid, set to default value to avoid error.
    if plot_type in ['Scatterplot', 'Barplot']:
        if x_axis_selection is None or x_axis_selection not in ['Released_Year', 'Runtime', 'Genre']:
            x_axis_selection = 'Released_Year'

    # Filter data by year, genre, and vote conditions
    filtered_data = api.filter_data(year_range, genre_selection, min_votes_slider)

    # If filtered data is empty (no data meets all conditions), return error message
    if filtered_data.empty:
        return pn.pane.Markdown('### No data found matching the selected criteria.')

    # Set border to black if border_checkbox is checked
    edgecolor = 'black' if border_checkbox else 'none'

    # Create plot based on plot_type selection
    if plot_type in ['Scatterplot', 'Barplot']:
        plot_figure = api.create_plot(plot_type, width, height, x_axis_selection, y_axis_selection, filtered_data,
                                        color_picker, edgecolor)
    elif plot_type == 'Histogram':
        plot_figure = api.create_plot(plot_type, width, height, y_axis = y_axis_selection, data = filtered_data,
                                      color = color_picker, edgecolor = edgecolor)

    # Get current axis
    ax = plt.gca()

    # If wanting to show all x-ticks, use get_xticks() for tick locations. Force x-tick locations to be integers.
    tick_locations = ax.get_xticks()
    int_tick_locations = np.arange(int(min(tick_locations)), int(max(tick_locations)) + 1)
    plt.xticks(ticks = int_tick_locations, fontsize = x_tick_font_size)

    # If not wanting to show all x-ticks, calculate new tick locations to skip tick_skip_slider values (documentation
    # help from ChatGPT)
    if not show_all_x_ticks and tick_skip_slider > 1:
        int_tick_locations = [loc for i, loc in enumerate(int_tick_locations) if i % tick_skip_slider == 0]
        plt.xticks(ticks = int_tick_locations, fontsize = x_tick_font_size)

    # Rotate x-ticks if tilt_x_ticks is checked
    if tilt_x_ticks:
        plt.xticks(rotation = 45)

    return pn.pane.Matplotlib(plot_figure)


# CALLBACK BINDINGS (Connecting widgets to callback functions)

# Bind datatable to widgets
datatable = pn.bind(generate_table, x_axis_selection, y_axis_selection, year_range, min_votes_slider,
                    genre_selection, include_series_checkbox, include_overview_checkbox)

# Bind plot to widgets
plot = pn.bind(generate_plot, plot_type, x_axis_selection, y_axis_selection, year_range, min_votes_slider,
               genre_selection, width, height, color_picker, border_checkbox, tilt_x_ticks, x_tick_font_size,
               show_all_x_ticks, tick_skip_slider)


# DASHBOARD WIDGET CONTAINERS ("CARDS")

card_width = 320

# Add headers above widgets to guide user selections
plot_type_header = pn.pane.Markdown("Plot type: ", margin = (-10, 0, -20, 0))
x_axis_selection_header = pn.pane.Markdown("#### For scatterplots and barplots only: ", margin = (0, 0, -10, 0))
y_axis_selection_header = pn.pane.Markdown("#### For all plot types: ", margin = (0, 0, -10, 0))
show_all_x_ticks_header = pn.pane.Markdown("#### If NOT showing all x-axis ticks: ", margin = (-5, 0, -15, 0))

# Create 'Search' card
search_card = pn.Card(
    pn.Column(
        plot_type_header,
        plot_type,
        x_axis_selection_header,
        x_axis_selection,
        y_axis_selection_header,
        y_axis_selection,
        year_range,
        min_votes_slider,
        genre_selection
    ),
    title = 'Search', width = card_width, collapsed = False
)

# Create 'Plot' card
plot_card = pn.Card(
    pn.Column(
        width,
        height,
        color_picker,
        border_checkbox,
        x_tick_font_size,
        tilt_x_ticks,
        show_all_x_ticks,
        show_all_x_ticks_header,
        tick_skip_slider
    ),
    title = 'Plot', width = card_width, collapsed = True
)

# Create 'Table' card
table_card = pn.Card(
    pn.Column(
        include_series_checkbox,
        include_overview_checkbox
    ),
    title = 'Table', width = card_width, collapsed = True
)


# LAYOUT

# Set up layout
layout = pn.template.FastListTemplate(
    title = 'IMDb Database Explorer',
    sidebar = [
        search_card,
        plot_card,
        table_card
    ],
    theme_toggle = False,
    main = [
        pn.Tabs(
            ('Plot', plot),  # Replace None with callback binding
            ('Table', datatable),  # Replace None with callback binding
            active = 0  # Which tab is active by default?
        )
    ],
    header_background = '#000000'
).servable()

layout.show()
