"""
file: lyricool_app.py

Description: Apply text analysis library to multiple lyrics.

"""

# import necessary packages
from lyricool import Lyricool, LyricoolParsingError
import lyricool_parsers as lp
import matplotlib.pyplot as plt
import pprint as pp


def main():

    try:
        # load in lyrics data for desired songs (I chose to pull the first song of each Beatles album (up to 10) to
        # analyze change in style over time)
        lyr = Lyricool()
        lyr.load_text('https://www.azlyrics.com/lyrics/beatles/isawherstandingthere.html',
                      "I Saw Her Standing There", parser = lp.az_lyrics_parser)
        lyr.load_text('https://www.azlyrics.com/lyrics/beatles/itwontbelong.html', "It Won't Be Long",
                       parser = lp.az_lyrics_parser)
        lyr.load_text('https://www.azlyrics.com/lyrics/beatles/aharddaysnight.html', "A Hard Day's Night",
                       parser = lp.az_lyrics_parser)
        lyr.load_text('https://www.azlyrics.com/lyrics/beatles/noreply.html', "No Reply",
                      parser = lp.az_lyrics_parser)
        lyr.load_text('https://www.azlyrics.com/lyrics/beatles/help.html', "Help!",
                      parser = lp.az_lyrics_parser)
        lyr.load_text('https://www.azlyrics.com/lyrics/beatles/drivemycar.html', "Drive My Car",
                       parser = lp.az_lyrics_parser)
        lyr.load_text('https://www.azlyrics.com/lyrics/beatles/taxman.html', "Taxman",
                      parser = lp.az_lyrics_parser)
        lyr.load_text('https://www.azlyrics.com/lyrics/beatles/sgtpepperslonelyheartsclubband.html',
                       "Sgt. Pepper's Lonely Hearts Club Band", parser = lp.az_lyrics_parser)
        lyr.load_text('https://www.azlyrics.com/lyrics/beatles/backintheussr.html',"Back in the USSR",
                      parser = lp.az_lyrics_parser)
        lyr.load_text('https://www.azlyrics.com/lyrics/beatles/cometogether.html', "Come Together",
                       parser = lp.az_lyrics_parser)

        # pretty-print lyrics data (optional)
        # pp.pprint(lyr.data)

        # generate three visualizations (need to call plt.show() in main function as opposed to individual functions
        # in order to generate each figure in its own window using TkAgg)
        lyr.wordcount_sankey()
        lyr.most_frequent_words_subplots(lyr.data)
        lyr.polarity_subjectivity_scatterplot(lyr.data)
        plt.show()

    # handle parsing-specific errors (used ChatGPT for error syntax)
    except LyricoolParsingError as e:
        print(f'Error: {e}')
        if e.filename:
            print(f'Occurred in file: {e.filename}')

    # handle missing files
    except FileNotFoundError as e:
        print(f'File not found: {getattr(e, 'filename', 'Unknown')}')

    # catch all other exceptions
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


if __name__ == '__main__':
    main()
