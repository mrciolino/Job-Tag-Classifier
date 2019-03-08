"""
Cutback.io 
Collection of data collection functions that imports
and cleans our data before feature engineering
"""

import pandas as pd


def load_data(data_file):
    df = pd.read_csv(data_file)
    return df


def remove_empty_rows(df):
    try:
        df = df[df["job_description"].notnull()]
        df = df[df["job_title"].notnull()]
    except:
        print("Unable to remove empty row from the dataframe")
        sys.exit(0)


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
