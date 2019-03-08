"""
Cutback.io
Collection of data processing functions that takes
our features converts them into a format usable by AI models
"""
import sys
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer


def remove_web_text(html):
    soup = BeautifulSoup(html, features="html.parser")
    for script in soup(["script", "style"]):  # remove all javascript and stylesheet code
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


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
    pat = r'\b(?:{})\b'.format('|'.join(stop_words))
    df.job_description = df.job_description.str.replace(pat, '')
    return df


def stem_text(df):
    def stem_sentences(sentence):
        tokens = sentence.split()
        stemmed_tokens = [parsing.stem_text(token) for token in tokens]
        return ' '.join(stemmed_tokens)
    df.job_description = df.job_description.apply(stem_sentences)
    return df


def target_encoder(df):
    # create job target encoder
    labeler = MultiLabelBinarizer()
    y = labeler.fit_transform(df.job_targets)
    # save target corpus
    with open("Models/Tokenizers/target_tokens.pkl", 'wb') as vocab_file:
        pickle.dump(labeler, vocab_file, protocol=pickle.HIGHEST_PROTOCOL)
    return y


def scale_pos_features(df):
    # seperate text feature numerical values
    text_df = df[['char_count', 'word_density', 'word_density', 'punctuation_count',
                  'upper_cgase_word_count', 'upper_case_word_count', 'stopword_count']]
    text_feature_matrix = text_df.values
    # scale text feature matrix
    scaler = MinMaxScaler()
    text_feature_matrix = scaler.fit_transform(text_feature_matrix)

    return text_feature_matrix


def collect_dataframes(matrix_1,matrix_2,matrix_3):
    # research locals().keys() and open ended arguments
    try:
        x = np.hstack((matrix_1,matrix_2,matrix_3))
    except:
        print("Unable to stack all of the matricies")
        sys.exit(0)

    return x


def hash_trick(df, num_words):

    # use hashing trick to allow new words to automatically be used in future data
    description_matrix = hashing_trick(df.job_description, num_words * 1.5)
    title_matrix = hashing_trick(df.job_title, num_words * 1.5)

    # return the data
    return description_matrix, title_matrix
