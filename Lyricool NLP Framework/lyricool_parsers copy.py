"""
file: lyricool_parsers.py

Description: A custom parser to pre-process and parse song lyrics from AZLyrics.com.
Final output of parser will be a dictionary containing wordcount, numwords, avg_word_length,
unique_word_count, type_token_ratio, sentiment, polarity, subjectivity, and emotions for given lyrics.

"""

# import necessary packages
import time
import requests
import string
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # (used ChatGPT to help manage huggingface warning)
from bs4 import BeautifulSoup
from collections import Counter
from textblob import TextBlob
from transformers import pipeline
from lyricool import Lyricool, LyricoolParsingError, STOPWORDS_FILE


def az_lyrics_preprocessor(url, stopwords_file=None):
    """ Pre-process lyrics from an AZLyrics URL for future parsing and visualization.
    Filter stopwords from stopwords file, if provided. """
    try:
        # fetch URL content (used ChatGPT for requests syntax) using delay to avoid triggering anti-scraping restrictions
        time.sleep(10)
        response = requests.get(url)
        response.raise_for_status()

        # parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # use CSS Selectors for nested lyrics class (consistent for all AZLyrics.com pages) (used ChatGPT for CSS syntax)
        selector = 'html > body.az-song-text > div.container.main-page > div.row > div.col-xs-12.col-lg-8.text-center'
        lyric_directory = soup.select_one(selector)
        if not lyric_directory:
            # if HTML structure is different, alert user
            raise LyricoolParsingError('Lyrics section not found in HTML:', filename = url)

        # given lyrics are in fifth div under col-xs-12 col-lg-8 text-center class, find target div containing lyrics
        # or raise error message if not found
        possible_lyric_divs = lyric_directory.find_all('div')
        if len(possible_lyric_divs) <= 5:
            raise LyricoolParsingError('Unexpected HTML structure: Unable to find target lyrics div in',
                                     filename = url)

        target_div = possible_lyric_divs[5]

        # loop through <br> tags in div containing lyrics and extract text. join all lyrics lines.
        lyrics_lines = []
        for br_tag in target_div.find_all('br'):
            # extract text from previous sibling (actual lyrics text) (used ChatGPT for sibling syntax)
            text = br_tag.previous_sibling
            if text:
                lyrics_lines.append(text.strip())
        raw_lyrics = '\n'.join(lyrics_lines)

        # clean lyrics (remove unnecessary whitespace, punctuation, and capitalization)
        cleaned_lyrics = []
        for char in raw_lyrics.lower():
            if char in string.ascii_lowercase or char.isdigit() or char.isspace():
                cleaned_lyrics.append(char)
        clean_lyrics = ''.join(cleaned_lyrics)

        # filter out stopwords
        stopwords = set()
        if stopwords_file:
            stopwords = Lyricool.load_stop_words(stopwords_file = STOPWORDS_FILE)
        filtered_lyrics = ' '.join(word for word in clean_lyrics.split() if word not in stopwords)
        return filtered_lyrics

    # handle errors using exception class
    except requests.exceptions.RequestException as e:
        raise LyricoolParsingError(f'Error fetching URL: {e}', filename = url)
    except Exception as e:
        raise LyricoolParsingError(f'Error processing lyrics: {e}', filename = url)


def az_lyrics_parser(url):
    """ Parse song lyrics from an AZLyrics URL and produce extracted data results in the form of a dictionary. """
    try:
        # pre-process lyrics from given URL
        filtered_lyrics = az_lyrics_preprocessor(url, stopwords_file = STOPWORDS_FILE)
        if not filtered_lyrics:
            raise LyricoolParsingError('No lyrics found after preprocessing.', filename = url)
        words = filtered_lyrics.split()

        # calculate word count
        wc = Counter(words)

        # calculate total words
        num = len(words)

        # calculate average word length
        total_characters = sum(len(word) for word in words)
        avg_word_length = total_characters / num if num > 0 else 0

        # calculate unique word count
        unique_word_count = len(set(words))

        # calculate type-token ratio (number of unique words / total number of words)
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if len(words) > 0 else 0

        # conduct sentiment analysis using TextBlob (used ChatGPT for TextBlob syntax)
        blob = TextBlob(filtered_lyrics)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        if polarity > 0:
            sentiment = 'Positive'
        elif polarity < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        # conduct emotion analysis using pipeline (used ChatGPT for pipeline syntax)
        classifier = pipeline('text-classification', model = 'j-hartmann/emotion-english-distilroberta-base',
                                device = 0)
        emotions = classifier(filtered_lyrics)
        emotion_dict = {item['label']: item['score'] for item in emotions}

        # return final output in format of default parser from lyricool.py
        return {
            'word_count': wc,
            'num_words': num,
            'avg_word_length': avg_word_length,
            'unique_word_count': unique_word_count,
            'type_token_ratio': ttr,
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'emotions': emotion_dict
        }

    # handle errors using exception class
    except LyricoolParsingError as e:
        # re-raise LyricoolParsingError for higher-level handling
        raise e
    except Exception as e:
        raise LyricoolParsingError(f'Unexpected error during parsing: {e}', filename = url)
