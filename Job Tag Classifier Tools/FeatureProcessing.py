"""
Matthew Ciolino - Job Tag Classifier
Collection of data processing functions that takes
our features converts them into a format usable by AI models
"""
import re
import sys
import pickle
import traceback
import numpy as np
from gensim import parsing
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
             'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
             'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
             'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
             'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
             'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
             'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
             'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
             'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
             "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
             "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
cList = {"ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have",
         "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
         "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
         "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
         "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not",
         "it'd": "it had", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
         "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
         "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
         "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
         "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
         "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
         "so's": "so is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there had",
         "there'd've": "there would have", "there's": "there is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
         "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we had",
         "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
         "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
         "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",
         "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
         "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
         "y'all": "you all", "y'alls": "you alls", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
         "y'all've": "you all have", "you'd": "you had", "you'd've": "you would have", "you'll": "you you will", "you'll've": "you you will have",
         "you're": "you are", "you've": "you have"}
c_re = re.compile('(%s)' % '|'.join(cList.keys()))


def remove_web_text(html):
    # create a new bs4 object from the html data loaded
    soup = BeautifulSoup(html, features="html.parser")
    for script in soup(["script", "style"]):  # remove all javascript and stylesheet code
        script.extract()
    text = soup.get_text()     # get text
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))  # break multi-headlines into a line each
    text = '\n'.join(chunk for chunk in chunks if chunk)    # drop blank lines
    return text


def expandcontractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


def clean_text(df):
    # parse html
    df.job_description = [remove_web_text(text) for text in df.job_description]
    # remove tags by regex
    df.job_description = df.job_description.str.replace('<[^>]+>', '')
    # repalce \r\n with a space
    df.job_description = df.job_description.str.replace("\r\n", ' ')
    return df


def strip_text(df):
    # expand contractions
    df.job_description = [expandcontractions(text) for text in df.job_description]
    # make lowercase
    df.job_description = [text.lower() for text in df.job_description]
    df.job_title = [text.lower() for text in df.job_title]
    # remove punctuation
    df.job_description = df.job_description.str.replace('[^\w\s]', '')
    df.job_title = df.job_title.str.replace('[^\w\s]', '')
    # remove digits
    df.job_description = df.job_description.str.replace('\d+', '')
    # remove stopwords
    pat = r'\b(?:{})\b'.format('|'.join(stopwords))
    df.job_description = df.job_description.str.replace(pat, '')
    return df


def stem_text(df):
    def stem_sentences(sentence):
        tokens = sentence.split()
        stemmed_tokens = [parsing.stem_text(token) for token in tokens]
        return ' '.join(stemmed_tokens)
    df.job_description = df.job_description.apply(stem_sentences)
    return df


def scale_pos_features(df):

    try:
        # seperate text feature numerical values
        text_df = df[['char_count', 'word_density', 'word_density', 'punctuation_count',
                      'upper_case_word_count', 'stopword_count', 'upper_case_word_count']]
        text_feature_matrix = text_df.values
        # scale text feature matrix
        scaler = MinMaxScaler()
        text_feature_matrix = scaler.fit_transform(text_feature_matrix)
    except:
        print("ERROR: Unable to scale the text features")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

    return text_feature_matrix


def hash_trick(df):

    def hash(text, num_words):
        vectorizer = HashingVectorizer(n_features=num_words, alternate_sign=False)
        return vectorizer.fit_transform(text).todense()

    try:
        # use hashing trick to allow new words to automatically be used in future data
        # the length of the hash table must be fixed throught training and predicition
        # if you want to change the length you must re train the model again
        # Collisions can be avoided using larger sized arrays but for now...
        description_matrix = hash(df.job_description, pow(2, 17))
        title_matrix = hash(df.job_title, pow(2, 13) - 7)
    except:
        print("ERROR: Unable to convert text with hashing trick")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

    # return the data
    return description_matrix, title_matrix


def collect_dataframes(matrix_1, matrix_2, matrix_3):
    # research locals().keys() and open ended arguments
    try:
        x = np.hstack((matrix_1, matrix_2, matrix_3))
    except:
        print("ERROR: Unable to stack all of the matricies")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

    return x


def target_encoder(df):

    try:
        # create job target encoder
        labeler = MultiLabelBinarizer()
        y = labeler.fit_transform(df.job_targets)
        # save target corpus
        with open("Models/Tokenizers/target_tokens.pkl", 'wb') as vocab_file:
            pickle.dump(labeler, vocab_file, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print("ERROR: Unable to one-hot-encode target")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

    return y


def feature_processing(df):

    try:
        df = clean_text(df)
        df = strip_text(df)
        df = stem_text(df)
    except:
        print("ERROR: Unable to preprocess text into desired format")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

    text_feature_matrix = scale_pos_features(df)
    description_matrix, title_matrix = hash_trick(df)

    x = collect_dataframes(description_matrix, title_matrix, text_feature_matrix)
    y = target_encoder(df)

    return x, y
