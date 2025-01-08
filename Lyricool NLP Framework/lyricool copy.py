"""
file: lyricool.py

Description: A reusable library for lyric analysis and comparison. In theory, the framework should support any lyrics
of interest, although a custom parser may be useful if pulling from website (such as AZLyrics.com).

"""

# import necessary packages
import random as rnd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use('TkAgg')  # use TkAgg as the backend for interactive plots (plots open in new window for clarity)
import matplotlib.pyplot as plt
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # (used ChatGPT to help manage huggingface warning)
import warnings
warnings.filterwarnings('ignore', message = 'The figure layout has changed to tight') # ignore this warning
from collections import defaultdict, Counter
from textblob import TextBlob
from transformers import pipeline
from math import ceil


# define global variables
STOPWORDS_FILE = 'stopwords.txt'


class Lyricool:

    def __init__(self):
        """ Constructor (ex: datakey --> (filelabel --> datavalue)). """
        self.data = defaultdict(dict)


    @staticmethod
    def load_stop_words(stopwords_file):
        """ Load in stopwords file and return list of stopwords for future filtering. """
        with open(stopwords_file, 'r') as file:
            stopwords = [line.strip().lower() for line in file if line.strip()]
            return stopwords


    def default_preprocessor(self, filename, stopwords_file=None):
        """ Pre-process standard text file of lyrics for future parsing and visualization.
         Filter stopwords from stopwords file, if provided. """
        try:
            # open and read lyrics file
            with open(filename, 'r') as file:
                text = file.read()
                words = text.split()

                # clean lyrics (remove unnecessary whitespace, punctuation, and capitalization) (used ChatGPT for char
                # syntax)
                cleaned_lyrics = []
                for word in words:
                    cleaned_word = ''.join(char for char in word if char.isalpha() or char.isdigit())
                    cleaned_lyrics.append(cleaned_word.lower())
                clean_lyrics = ' '.join(cleaned_lyrics)

                # filter out stopwords
                stopwords = set()
                if stopwords_file:
                    stopwords = self.load_stop_words(stopwords_file)
                filtered_lyrics = ' '.join(word for word in clean_lyrics.split() if word not in stopwords)
            return text, filtered_lyrics

        # handle errors using exception class
        except FileNotFoundError:
            raise LyricoolParsingError(f'File not found: {filename}', filename)
        except Exception as e:
            raise LyricoolParsingError(f'An error occurred while preprocessing the file: {e}', filename)


    def default_parser(self, filename):
        """ Parse standard text file of lyrics and produce extracted data results in the form of a dictionary. """
        try:
            results = {
                'word_count': Counter(),
                'num_words': rnd.randrange(10, 50),
                'avg_word_length': 0,
                'unique_word_count': 0,
                'type_token_ratio': 0,
                'sentiment': '',
                'polarity': 0,
                'subjectivity': 0,
                'emotions': {},
            }

            # pre-process text (lyrics) from given filename
            text, filtered_lyrics = self.default_preprocessor(filename, STOPWORDS_FILE)
            words = filtered_lyrics.split()

            # calculate word count
            results['word_count'] = Counter(words)

            # calculate total words
            results['num_words'] = len(words)

            # calculate average word length
            total_characters = sum(len(word) for word in words)
            results['avg_word_length'] = total_characters / len(words) if len(words) > 0 else 0

            # calculate unique word count
            results['unique_word_count'] = len(set(words))

            # calculate type-token ratio (number of unique words / total number of words)
            ttr = len(set(words)) / len(words) if len(words) > 0 else 0
            results['type_token_ratio'] = ttr

            # conduct sentiment analysis using TextBlob (used ChatGPT for TextBlob syntax)
            blob = TextBlob(text)
            results['polarity'] = blob.sentiment.polarity
            results['subjectivity'] = blob.sentiment.subjectivity
            if blob.sentiment.polarity > 0:
                results['sentiment'] = 'Positive'
            elif blob.sentiment.polarity < 0:
                results['sentiment'] = 'Negative'
            else:
                results['sentiment'] = 'Neutral'

            # conduct emotion analysis using pipeline (used ChatGPT for pipeline syntax)
            classifier = pipeline('text-classification', model = 'j-hartmann/emotion-english-distilroberta-base',
                                    device = 0)
            emotions = classifier(text)
            results['emotions'] = {item['label']: item['score'] for item in emotions}
            return results

        # handle errors using exception class
        except LyricoolParsingError:
            # re-raise custom exceptions
            raise
        except Exception as e:
            raise LyricoolParsingError(f'An error occurred while parsing the file: {e}', filename)


    def load_text(self, filename, label=None, parser=None):
        """ Register a document (lyrics) with the framework. Extract and store data to be used later by
        the visualizations. """
        # if no parser is given, use default. if parser is provided, use custom parser on given lyrics file.
        if parser is None:
            results = self.default_parser(filename)
        else:
            results = parser(filename)

        # if no label provided, use provided filename
        if label is None:
            label = filename

        # add lyrics results to self.data dictionary
        for k, v in results.items():
            self.data[k][label] = v


    def wordcount_sankey(self, word_list=None, k=5, **kwargs):
        """ Map each text (lyrics) to words using a Sankey diagram, where the thickness of the line
        is the number of times that word occurs in the text. Users can specify a particular set of words,
        or the words can be the union of the k most common words across each text file (excluding stop words). """
        # if word_list is not provided, use union of k most common words across texts (lyrics) (used ChatGPT for
        # most_common syntax)
        if word_list is None:
            word_list = set()
            for word_counts in self.data['word_count'].values():
                word_list.update(word for word, _ in word_counts.most_common(k))

        # ensure no words provided in user-specified word_list are missing from all lyrics. if there are missing
        # words, alert user in print statement.
        missing_list = []
        for word in word_list:
            if not any(word in word_counts for word_counts in self.data['word_count'].values()):
                missing_list.append(word)
        if missing_list:
            print(f'The following word(s) are not present in any lyrics files: {', '.join(missing_list)}')

        # create list to filter data in Sankey diagram format (source, target, value)
        sankey_data = []

        # create empty set to track which words are present in text (lyric) word counts
        words_in_text = set()

        # accumulate word counts across all given text labels for each word in word_list
        for text_label, word_counts in self.data['word_count'].items():
            for word in word_list:
                # default to 0 if word is not in current text label
                count = word_counts.get(word, 0)
                if count > 0:
                    sankey_data.append({'source': text_label, 'target': word, 'value': count})
                    # track words that are actually found in lyrics
                    words_in_text.add(word)

        # add words that are in word_list but not in any lyrics
        for word in word_list:
            if word not in words_in_text:
                # create separate source titled 'Missing words' to clearly show missing words not in any lyrics
                sankey_data.append({'source': 'Missing words', 'target': word, 'value': 1})

        # convert Sankey data to dataframe
        sankey_df = pd.DataFrame(sankey_data)

        # map labels in 'source' and 'target' to integer codes (used code from hw2)
        def code_mapping(df, src, targ):
            # get distinct codes
            labels = sorted(set(df[src].astype(str).tolist() + df[targ].astype(str).tolist()))
            # create a label -> code mapping
            label_map = {label: idx for idx, label in enumerate(labels)}
            # substitute codes for labels in dataframe (will map src and targ separately in case of different data types)
            df[src] = df[src].map(label_map)
            df[targ] = df[targ].map(label_map)
            return df, labels

        df, labels = code_mapping(sankey_df, 'source', 'target')

        # create links for Sankey diagram
        link = {'source': df['source'], 'target': df['target'], 'value': df['value']}

        # modify Sankey using kwargs (if provided). otherwise, set to default values.
        thickness = kwargs.get('thickness', 70)
        pad = kwargs.get('pad', 50)

        # create nodes for Sankey diagram
        node = {'label': labels, 'thickness': thickness, 'pad': pad}

        # create Sankey diagram (used ChatGPT for title syntax)
        sk = go.Sankey(link = link, node = node)
        fig = go.Figure(sk)
        fig.update_layout(title_text = 'Song-to-Word Sankey Diagram', font_size = 10, title_x = 0.5)
        fig.show(renderer = 'browser')


    def most_frequent_words_subplots(self, lyric_data, selected_labels=None, n_most_frequent=10, cols=3):
        """ Determine n-most frequent words in each text file (lyrics) and create subplots of n-most frequent words
        for each file in selected_labels (one sub-plot for each lyrics set). """
        # extract word count data
        word_count_data = lyric_data.get('word_count', {})

        # check for invalid labels. if they exist, alert user they will not be plotted.
        if selected_labels:
            invalid_labels = [label for label in selected_labels if label not in word_count_data]
            if invalid_labels:
                print(f'The following labels do not exist in the lyrics data and will not be plotted: ' 
                        f'{', '.join(invalid_labels)}')

        # filter word counts based on selected_labels (if none, use all in lyric data)
        if selected_labels:
            filtered_data = {label: word_count_data[label] for label in selected_labels if label in word_count_data}
        else:
            filtered_data = word_count_data

        # if all selected_labels are invalid, num_files will be 0 and no plots will be created. alert user if so.
        num_files = len(filtered_data)
        if num_files == 0:
            print('None of the given labels exist in the lyrics data. No plot will be generated.')
            return

        # calculate grid dimensions based on user-input columns (default is 3) and adjust figure size by subplots
        rows = ceil(num_files / cols)
        fig, axes = plt.subplots(rows, cols, figsize = (cols * 6, rows * 5), constrained_layout = True)

        # flatten axes for easier indexing (used ChatGPT for syntax)
        axes = axes.flatten()

        # generate a colormap (tab10) to pick unique colors for each subplot
        subplot_colors = plt.cm.tab10(np.linspace(0, 1, num_files))

        # plot each text's (lyrics') word counts for n_most_frequent words using subplot bar plots
        for idx, (text_label, word_counts) in enumerate(filtered_data.items()):
            most_common_words = word_counts.most_common(n_most_frequent)
            # if no most_common_words, set (words, counts) to empty lists (used ChatGPT to help zip common words)
            words, counts = zip(*most_common_words) if most_common_words else ([], [])

            ax = axes[idx]
            ax.bar(words, counts, color = subplot_colors[idx])
            ax.set_xlabel('Word', fontsize = 9)
            ax.set_ylabel('Frequency', fontsize = 9)
            ax.tick_params(axis = 'x', labelsize = 7)
            ax.tick_params(axis = 'y', labelsize = 7)
            ax.set_title(f'Top {n_most_frequent} Words in {text_label}', fontsize = 10)

        # remove extra axes based on extra space in figure
        for ax in axes[num_files:]:
            ax.axis('off')

        # add an overall title to figure
        fig.suptitle('Subplots of Most Frequent Words in Selected Songs', fontsize = 12)


    def polarity_subjectivity_scatterplot(self, lyric_data, selected_labels=None):
        """ Create a scatterplot comparing polarity vs. subjectivity of each text file (lyrics) in selected_labels. """
        # extract polarity and subjectivity data
        polarity_data = lyric_data.get('polarity', {})
        subjectivity_data = lyric_data.get('subjectivity', {})

        # filter polarity and subjectivity data based on selected_labels. if invalid labels exist, alert user they
        # will not be plotted.
        if selected_labels:
            valid_labels = [label for label in selected_labels if label in polarity_data and label in subjectivity_data]
            invalid_labels = set(selected_labels) - set(valid_labels)
            if invalid_labels:
                print(f'The following labels do not exist in lyrics data and will not be plotted: '
                        f'{', '.join(invalid_labels)}')
            filtered_data = {label: {'polarity': polarity_data[label], 'subjectivity': subjectivity_data[label]}
                             for label in valid_labels}
        else:
            # if no selected_labels, use all polarity and subjectivity data in lyric data
            filtered_data = {label: {'polarity': polarity_data[label], 'subjectivity': subjectivity_data[label]}
                             for label in polarity_data if label in subjectivity_data}

        # if no valid polarity/subjectivity data, alert user
        if not filtered_data:
            print('No polarity/subjectivity data to plot.')
            return

        # prepare data for scatter plot
        polarities = []
        subjectivities = []
        labels = list(filtered_data.keys())

        for label, data in filtered_data.items():
            polarity = data['polarity']
            subjectivity = data['subjectivity']
            polarities.append(polarity)
            subjectivities.append(subjectivity)

        # define unique colors for each text (lyrics) using colormap (tab20)
        num_texts = len(filtered_data)
        text_colors = plt.cm.get_cmap('tab20', num_texts)

        # create scatterplot
        plt.figure(figsize = (10, 6))

        # plot each text (lyrics set) using its respective color
        for i, label in enumerate(labels):
            plt.scatter(polarities[i], subjectivities[i], color = text_colors(i), label = label, s = 100,
                        edgecolor = 'black')
        plt.xlabel('Polarity')
        plt.ylabel('Subjectivity')
        # strictly define x- and y-limits because they will always be the same for polarity and subjectivity
        plt.xlim(-1, 1)
        plt.ylim(0, 1)
        plt.legend(title = 'Selected Songs', bbox_to_anchor = (1.05, 1), loc = 'upper left')
        plt.title('Polarity vs. Subjectivity Across Selected Songs')
        plt.tight_layout()


class LyricoolParsingError(Exception):
    """ Custom exception for handling errors in parsing lyrics. (used ChatGPT for Exception syntax) """
    def __init__(self, message, filename=None):
        super().__init__(message)
        self.filename = filename
