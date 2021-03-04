from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
wordnet_tags = ['n', 'v']
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]
stemmer = PorterStemmer()
print('Stemmed:', [[stemmer.stem(token) for token in
word_tokenize(document)] for document in corpus])
def lemmatize(token, tag):
    if tag[0].lower() in ['n', 'v']:
        return lemmatizer.lemmatize(token, tag[0].lower())
    return token
lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in
corpus]
print(tagged_corpus)
print('Lemmatized:', [[lemmatize(token, tag) for token, tag in
document] for document in tagged_corpus])




# from sklearn.feature_extraction.text import CountVectorizer
# corpus = [
#     'He ate the sandwiches',
#     'Every sandwich was eaten by him'
# ]
# vectorizer = CountVectorizer(binary=True, stop_words='english')
# print(vectorizer.fit_transform(corpus).todense())
# print(vectorizer.vocabulary_)
#
# corpus = [
#     'I am gathering ingredients for the sandwich.',
#     'There were many wizards at the gathering.'
# ]
#
# from nltk.stem.wordnet import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize('gathering', 'v'))
# print(lemmatizer.lemmatize('gathering', 'n'))
#
# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
# print(stemmer.stem('gathering'))
