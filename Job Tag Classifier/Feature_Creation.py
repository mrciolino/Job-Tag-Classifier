"""
Cutback.io
Collection of feature engineering functions that takes
creates features and prepares our data for the AI model
"""
import string


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


def pos_features(df):
    # find numerical data from text (i.e. number of stopwords)
    text_data_df = text_features(df)
    # find numerical word selection data from text (i.e. number of verbs)
    text_data_df = word_selection_features(df)
    # return data
    return text_data_df


def collect_job_tags(df):
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

def load_target_tokenizer():
    pass

def encode_target():
    pass

    
