import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import nltk
import pandas as pd
pd.options.mode.chained_assignment = None


pos_map = {
    'CC': 'n', 'CD': 'n', 'DT': 'n', 'EX': 'n', 'FW': 'n', 'IN': 'n', 'JJ': 'a', 'JJR': 'a', 'JJS': 'a', 'LS': 'n',
    'MD': 'v', 'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n', 'PDT': 'n', 'POS': 'n', 'PRP': 'n', 'PRP$': 'r',
    'RB': 'r',
    'RBR': 'r', 'RBS': 'r', 'RP': 'n', 'TO': 'n', 'UH': 'n', 'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v',
    'VBZ': 'v', 'WDT': 'n', 'WP': 'n', 'WP$': 'n', 'WRB': 'r'
}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def tokenizer(example_sent):
    # Capitalization
    example_sent = str(example_sent).lower()
    # HTML TAGS
    example_sent = BeautifulSoup(example_sent, 'lxml').text
    # EMAIL ADDRESSES
    example_sent = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', example_sent)
    # URLs
    example_sent = re.sub(r'http\S+', '', example_sent)
    # Punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(example_sent)
    # POS Tagging
    tags = nltk.pos_tag(word_tokens)
    # Lemmatization
    for i, word in enumerate(word_tokens):
        word_tokens[i] = lemmatizer.lemmatize(word, pos=pos_map.get(tags[i][1], 'n'))

    # stop words
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    # digits
    filtered_sentence = [w for w in filtered_sentence if not w.isdigit()]

    return filtered_sentence


# Elimina los títulos con valor "notitle"
def delete_notitle(title):
    if title == 'notitle':
        return ' '
    else:
        return title


def filter_dataset(dataframe):
    # Eliminamos todas las noticias que no están en inglés
    dataframe = dataframe.loc[(dataframe['language'] == 'english') | (dataframe['language'].isna())]

    # Aplicamos la función delete_notitle
    dataframe['title'] = dataframe['title'].apply(delete_notitle)

    # Nos quedamos solamente con las columnas title, text y type
    dataframe = dataframe.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'author', 'comments', 'country',
                          'crawled', 'domain_rank', 'id', 'language', 'likes', 'main_img_url',
                          'ord_in_thread', 'participants_count', 'published', 'replies_count',
                          'shares', 'site_url', 'spam_score', 'thread_title',
                          'uuid', 'caps_title', 'caps_thread', 'caps_text', 'title_len',
                          'thread_len', 'text_len', 'excl_title', 'excl_thread', 'excl_text',
                          'first_title', 'first_thread', 'first_text', 'second_title',
                          'second_thread', 'second_text', 'third_title', 'third_thread',
                          'third_text', 'anger', 'anticipation', 'disgust', 'fear', 'joy',
                          'sadness', 'surprise', 'trust', 'negative', 'positive', 'first_all',
                          'second_all', 'third_all'])

    # Unimos el texto y el título en un solo campo
    dataframe['text'] = dataframe['title'] + ' ' + dataframe['text']

    # Eliminamos la columna title
    dataframe = dataframe.drop(columns=['title'])

    # Mapeamos los valores real y fake como 0 y 1 respectivamente
    dataframe.loc[dataframe['type'] == 'fake', 'type'] = 1.
    dataframe.loc[dataframe['type'] == 'real', 'type'] = 0.

    return dataframe