"""
Matthew Ciolino - Job Tag Classifier 
Collection of feature engineering functions that takes
creates features and prepares our data for the AI model
"""
from textblob import TextBlob
import traceback
import string
import sys

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

def text_features(df):

    # Find number of words, stopwords, characters, word_density, punctuation count, uppercase words
    punctuation = string.punctuation
    df['char_count'] = df.job_description.apply(len)
    df['word_count'] = df.job_description.apply(lambda x: len(x.split()))
    df['word_density'] = df['char_count'] / (df['word_count'] + 1)
    df['punctuation_count'] = df.job_description.apply(lambda x: len("".join(_ for _ in x if _ in punctuation)))
    df['upper_case_word_count'] = df.job_description.apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    df['stopword_count'] = df.job_description.apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stopwords]))

    return df


def pos_check(text, flag):
    pos_dic = {'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
               'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
               'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
               'adj':  ['JJ', 'JJR', 'JJS'],
               'adv': ['RB', 'RBR', 'RBS', 'WRB']}
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tupple in wiki.tags:
            part_of_speech = list(tupple)[1]
            if part_of_speech in pos_dic[flag]:
                cnt += 1
    except:
        pass
    return cnt


def word_selection_features(df):

    # find number of different types of words in a job description
    df['noun_count'] = df.job_description.apply(lambda x: pos_check(x, 'noun'))
    df['verb_count'] = df.job_description.apply(lambda x: pos_check(x, 'verb'))
    df['adj_count'] = df.job_description.apply(lambda x: pos_check(x, 'adj'))
    df['adv_count'] = df.job_description.apply(lambda x: pos_check(x, 'adv'))
    df['pron_count'] = df.job_description.apply(lambda x: pos_check(x, 'pron'))

    return df


def pos_features(df):

    try:
        # find numerical data from text (i.e. number of stopwords)
        text_data_df = text_features(df)
    except:
        print("ERROR: Unable to create numerical text features from the data")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

    try:
        # find part of speech data from text (i.e. number of verbs)
        text_data_df = word_selection_features(df)
    except:
        print("ERROR: Unable to create part of speech features from the data")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

    # return data
    return text_data_df


def aggregate_job_tag_rows(df):
    # make place holder column
    df['job_targets'] = "N/A"
    # for each job id add every job tag that is listed for it across the rows
    for id in set(df.job_id):
        group = df.loc[df['job_id'] == id]
        target = group.job_tag_name.values
        df.at[group.index[0], 'job_targets'] = target.tolist()
    # get rid of all repeated job ids
    df = df[df.job_targets != "N/A"]
    return df


def feature_creation(df):

    df = pos_features(df)

    try:
        df = aggregate_job_tag_rows(df)
    except:
        print("ERROR: Unable to consoladate job tags")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

    return df
