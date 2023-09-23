import re
import nltk 
from tqdm import tqdm
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

def get_tokens_from_plaintext(text, pattern=None, stemming=True, lemmatizing=False):
    """
        This is a user defined function for cleaning the text data in a plain text.

        PARAMETERS
            text (str): It will be your corpus
            pattern (str): Provide a regex pattern of your choice
            stemming (bool): Provide 'True' if you want stemming otherwise make it 'False'
            lemmatizing (bool): Provide 'True' if you want lemmatization otherwise make it 'False'

        NOTE: If you want both stemming and lemmatization, then it will also 
        be possible by passting 'True' to both values.

        RETURN:
            Return type is a 'Tuple': (Stemming_Corpus, Lemmatization_Corpus)
            You can unbind it like this:
                WITHOUT PATTERN:
                stem_corp, lemm_corp = get_tokens(text, pattern=None, stemming=True, lemmatizing=True)
                
                WITH PATTERN:
                stem_corp, lemm_corp = get_tokens(text, pattern='[^a-zA-Z]', stemming=True, lemmatizing=True)
    """
    ps = PorterStemmer()
    wn = WordNetLemmatizer()
    
    sentences = nltk.sent_tokenize(text)
    stem_corpus = []
    lem_corpus = []
    
    for i in tqdm(range(len(sentences))):
        review = sentences[i]
        if pattern is not None:
            review = re.sub(pattern, ' ', sentences[i])

        words = review.lower().split()
        
        if stemming:
            review = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            if len(review) > 2:
                stem_corpus.append(review)
        
        if lemmatizing:
            review = [wn.lemmatize(word) for word in words if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            if len(review) > 2:
                lem_corpus.append(review)

    return stem_corpus, lem_corpus

def get_tokens_from_dataframe(df, colname='', pattern=None, stemming=True, lemmatizing=False):
    """
        This is a user defined function for cleaning the text data in a dataframe.

        PARAMETERS
            df (dataframe): It will be your desired dataframe
            colname (series): It will be your desired column want to process
            pattern (str): Provide a regex pattern of your choice
            stemming (bool): Provide 'True' if you want stemming otherwise make it 'False'
            lemmatizing (bool): Provide 'True' if you want lemmatization otherwise make it 'False'

        NOTE: If you want both stemming and lemmatization, then it will also 
        be possible by passting 'True' to both values.

        RETURN:
            Return type is a 'Tuple': (Stemming_Corpus, Lemmatization_Corpus)
            You can unbind it like this:
                WITHOUT PATTERN:
                stem_corp, lemm_corp = get_tokens(text, pattern=None, stemming=True, lemmatizing=True)
                
                WITH PATTERN:
                stem_corp, lemm_corp = get_tokens(text, pattern='[^a-zA-Z]', stemming=True, lemmatizing=True)
    """
    ps = PorterStemmer()
    wn = WordNetLemmatizer()
    
    stem_corpus = []
    lem_corpus = []
    
    for i in tqdm(range(len(df))):
        review = df[colname][i]
        if pattern is not None:
            review = re.sub(pattern, ' ', review)

        words = review.lower().split()
        
        if stemming:
            review = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            if len(review) > 2:
                stem_corpus.append(review)
        
        if lemmatizing:
            review = [wn.lemmatize(word) for word in words if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            if len(review) > 2:
                lem_corpus.append(review)

    return stem_corpus, lem_corpus

def get_tokens_from_column(text):
    """
        This is a user defined function for cleaning the text data of a column in a dataframe.

        PARAMETERS
            text (column): It will be your desired column want to process

        RETURN:
            Return type is a 'str': String
    """
    wn = WordNetLemmatizer()
    
    review = text.lower()
    review = re.sub(r"(?:@\w+|#\w+|https?://\S+|\d+|[^\w\s])", ' ', review)
    words = review.split()
    review = [wn.lemmatize(word) for word in words if not word in set(stopwords.words('english'))]

    return ' '.join(review)