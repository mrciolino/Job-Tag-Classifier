from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.text import hashing_trick, text_to_word_sequence
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
# from textblob import TextBlob
from gensim import parsing
import pandas as pd
import numpy as np
import pickle
import random
import string
import sys
import re


# list of directories for models and such
if sys.platform == "win32":
    data_file = "E:\DATA\Cutback\job_data.csv"
    processed_data_file = "E:\DATA\Cutback\processed_data.pkl"
    new_data_file = "E:\DATA\Cutback/new_job.csv"
else:
    data_file = "/Volumes/SD.Card/ML_Data/Cutback/job_data.csv"
    processed_data_file = "/Volumes/SD.Card/ML_Data/Cutback/processed_data.pkl"
    new_data_file = "/Volumes/SD.Card/ML_Data/Cutback/new_job.csv"

model_file = "job_tag_classifier/model/model.h5"
feature_corpus_file = "job_tag_classifier/model/full_corpus.pickle"
target_corpus_file = "job_tag_classifier/model/target_corpus.pickle"


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
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
              'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "i\
              t's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'tha\
              t', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'hav\
              ing', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', '\
              of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'abov\
              e', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'onc\
              e', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 's\
              ome', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just\
              ', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'cou\
              ldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn\
              ', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "should\
              n't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
pos_dic = {'noun': ['NN', 'NNS', 'NNP', 'NNPS'], 'pron': ['PRP', 'PRP$', 'WP', 'WP$'], 'verb': [
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], 'adj':  ['JJ', 'JJR', 'JJS'], 'adv': ['RB', 'RBR', 'RBS', 'WRB']}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))
global num_words


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


def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [parsing.stem_text(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def text_preprocess(df, stem, reduce, modify):

    # Remove any empty rows of data
    df = df[df["job_description"].notnull()]
    df = df[df["job_title"].notnull()]

    # parse html
    df.job_description = [remove_web_text(text) for text in df.job_description]
    # remove tags by regex
    df.job_description = df.job_description.str.replace('<[^>]+>', '')
    # repalce \r\n with a space
    df.job_description = df.job_description.str.replace("\r\n", ' ')

    # modify the text if wanted
    if modify:
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
        pat = r'\b(?:{})\b'.format('|'.join(stop_words))
        df.job_description = df.job_description.str.replace(pat, '')

    # reduce the amount of words we have by looking for certain key words
    if reduce:
        df.job_description = [reduce_job_description(text) for text in df.job_description]

    # stem the words
    if stem:
        df.job_description = df.job_description.apply(stem_sentences)

    # return the modified dataframe
    return df


def find_max_words(df, column):

    count = df[column].str.split().str.len()
    count.index = count.index.astype(str) + ' words:'
    count.sort_index(inplace=True)
    max(count)

    return max(count)


def get_job_tag_index(df, tag, outliers):

    # select the index of the job ids
    tags = df.job_tag_name.value_counts().index.tolist()
    if tag not in list(tags):
        print("Tag '%s' not in list of tags:" % (tag), tags)
        sys.exit(0)
    if tag == "all":
        tag = [df.job_tag_name.info]

    # get all the jobs tags from the tag_id
    job_ids = df.index[df['job_tag_name'] == tag].tolist()

    # add outliers to the data
    outlier_job_ids = []
    for num in range(outliers):
        outlier_job_ids.append(random.choice(df.index) - 1)

    # combine the job _ids and return
    return (job_ids + outlier_job_ids)


def pos_check(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_dic[flag]:
                cnt += 1
    except:
        pass
    return cnt


def text_features(text_data_df):

    # Find number of words, stopwords, characters, word_density, punctuation count, uppercase words
    punctuation = string.punctuation
    text_data_df['char_count'] = text_data_df.job_description.apply(len)
    text_data_df['word_count'] = text_data_df.job_description.apply(lambda x: len(x.split()))
    text_data_df['word_density'] = text_data_df['char_count'] / (text_data_df['word_count'] + 1)
    text_data_df['punctuation_count'] = text_data_df.job_description.apply(lambda x: len("".join(_ for _ in x if _ in punctuation)))
    text_data_df['upper_case_word_count'] = text_data_df.job_description.apply(
        lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    text_data_df['stopword_count'] = text_data_df.job_description.apply(
        lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words]))

    return text_data_df


def word_selection_features(text_data_df):

    # find number of different types of words in a job description
    text_data_df['noun_count'] = text_data_df.job_description.apply(lambda x: pos_check(x, 'noun'))
    text_data_df['verb_count'] = text_data_df.job_description.apply(lambda x: pos_check(x, 'verb'))
    text_data_df['adj_count'] = text_data_df.job_description.apply(lambda x: pos_check(x, 'adj'))
    text_data_df['adv_count'] = text_data_df.job_description.apply(lambda x: pos_check(x, 'adv'))
    text_data_df['pron_count'] = text_data_df.job_description.apply(lambda x: pos_check(x, 'pron'))

    return text_data_df


def feature_engineer(df, pos=False):

    # find numerical data from text (i.e. number of stopwords)
    text_data_df = text_features(df)
    # find numerical word selection data from text (i.e. number of verbs)
    if pos == True:
        text_data_df = word_selection_features(df)
    # return data
    return text_data_df


def aggregate_job_tags(df):
    df['job_targets'] = "N/A"
    for id in set(df.job_id):
        group = df.loc[df['job_id'] == id]
        target = group.job_tag_name.values
        df.at[group.index[0], 'job_targets'] = target.tolist()
    df = df[df.job_targets != "N/A"]
    return df


def df_to_values(df, preprocess):

    if preprocess:
        df = text_preprocess(df, stem=True, reduce=False, modify=True)

    # create job target encoder
    labeler = MultiLabelBinarizer()
    y = labeler.fit_transform(df.job_targets)
    # save target corpus
    with open(target_corpus_file, 'wb') as vocab_file:
        pickle.dump(labeler, vocab_file, protocol=pickle.HIGHEST_PROTOCOL)

    # use hashing trick to allow new words to automatically be used in future data
    num_words = len(set(text_to_word_sequence(text)))
    description_matrix = hashing_trick(df.job_description, num_words * 1.5)
    title_matrix = hashing_trick(df.job_title, num_words * 1.5)

    # seperate text feature numerical values
    text_df = df[['char_count', 'word_density', 'word_density', 'punctuation_count',
                  'upper_cgase_word_count', 'upper_case_word_count', 'stopword_count']]
    text_feature_matrix = text_df.values
    # scale text feature matrix
    scaler = MinMaxScaler()
    text_feature_matrix = scaler.fit_transform(text_feature_matrix)

    # join the datasets by row to get data
    x = np.hstack((title_matrix, description_matrix, text_feature_matrix))

    # return the data
    return x, y


def process_new_data(df, tokenizer):
    # clean up the text in the description using part of the text_preprocess function
    df = text_preprocess(df, stem=False, reduce=False, modify=False)

    # feature engineer
    df = feature_engineer(df, pos=False)

    # process text
    df = text_preprocess(df, stem=True, reduce=False, modify=True)

    # convert data into numbers
    description_matrix = hashing_trick(df.job_description, num_words * 1.5)
    title_matrix = hashing_trick(df.job_title, num_words * 1.5)

    # change this to only identify text matrix columns
    text_df = df[['char_count', 'word_density', 'word_density', 'punctuation_count',
                  'upper_case_word_count', 'upper_case_word_count', 'stopword_count']]
    text_feature_matrix = text_df.values
    # scale text feature matrix
    scaler = MinMaxScaler()
    text_feature_matrix = scaler.fit_transform(text_feature_matrix)

    # join the datasets by row to get data
    x = np.hstack((title_matrix, description_matrix, text_feature_matrix))

    return x


def similiarity_analysis(data):
    # Cosine Similiarity
    similarity_matrix = cosine_similarity(data)
    similarity_df = pd.DataFrame(similarity_matrix)

    # Linkage
    Z = linkage(similarity_matrix, 'ward')
    link = pd.DataFrame(Z, columns=['Document\Cluster 1', 'Document\Cluster 2', 'Distance', 'Cluster Size'], dtype='object')

    # Latent Dirichlet Allocation
    lda = LatentDirichletAllocation(n_components=2, max_iter=1000, random_state=0)
    dt_matrix = lda.fit_transform(data)
    features = pd.DataFrame(dt_matrix, columns=['Topic 1', 'Topic 2'])

    return features, link, similarity_df
