#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ! pip install nltk


# In[18]:


corpus="""
Hello I am arpita singh please do like my videos and.
subscribe my youtube channel.
"""


# In[19]:


#from paragraph to senetence...
from nltk.tokenize import sent_tokenize


# In[20]:


sent_tokenize(corpus)


# In[21]:


#from paragraph to words
from nltk.tokenize import word_tokenize


# In[22]:


word_tokenize(corpus)


# In[24]:


from nltk.tokenize import wordpunct_tokenize


# In[25]:


wordpunct_tokenize(corpus)


# In[26]:


from nltk.tokenize import TreebankWordTokenizer


# In[28]:


Tokenizer=TreebankWordTokenizer()


# In[30]:


Tokenizer.tokenize(corpus)


# # Stemming

# In[31]:


from nltk.stem import PorterStemmer


# In[32]:


stemming=PorterStemmer()


# In[33]:


words=['eating','eats','eaten','write','writing','written','wrote','programming','programs','programmed']


# In[34]:


for word in words:
    print(word+'----->'+stemming.stem(word))


# In[35]:


stemming.stem('congratulations')


# In[36]:


stemming.stem('History')


# In[37]:


#Lancaster stemming
from nltk.stem import LancasterStemmer


# In[38]:


lancaster=LancasterStemmer()


# In[39]:


for word in words:
    print(word+'---->'+lancaster.stem(word))


# # Regexpstemmer class

# In[40]:


from nltk.stem import RegexpStemmer


# In[41]:


reg_stemmer=RegexpStemmer('ing$|s$|e$|able$',min=4)


# In[42]:


reg_stemmer.stem('eating')


# In[45]:


reg_stemmer.stem("playing")


# # Snowballstemmer

# In[46]:


from nltk.stem import SnowballStemmer


# In[48]:


snowballstemmer= SnowballStemmer('english',ignore_stopwords=False)


# In[50]:


for word in words:
    print(word+'---->'+snowballstemmer.stem(word))


# In[52]:


stemming.stem('fairly'),stemming.stem('sportingly')


# In[54]:


snowballstemmer.stem('fairly'),snowballstemmer.stem('sportingly')


# # Lemmatization

# In[55]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


# In[57]:


for word in words:
    print(word+'---->'+lemmatizer.lemmatize(word))


# In[61]:


lemmatizer.lemmatize('better',pos='n')


# In[63]:


paragraphs="""
OSG is a leading manufacturer of taps, end mills, drills, and indexable cutting tools. OSG’s extensive line of high technology cutting tools features exclusive metallurgy, cutting geometries and proprietary surface treatments to help increase productivity, reliability and tool life. OSG also serves the fastener industry by offering a complete line of thread rolling, flat, cylindrical, planetary, rack and trim dies. The company markets its products to numerous industries including automotive, die mold, aerospace, energy, heavy industry and dental.
"""


# In[64]:


paragraphs


# In[65]:


from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# In[69]:


#tokenize ---- convert paragraph to sentences
nltk.download('punkt')
sentences=nltk.sent_tokenize(paragraphs)


# In[70]:


print(sentences)


# In[71]:


type(sentences)


# In[72]:


stemmer=PorterStemmer()


# In[77]:


stemmer.stem('drinking'),stemmer.stem('thinking')


# In[78]:


lemmatizer=WordNetLemmatizer()


# In[80]:


lemmatizer.lemmatize('goes')


# In[83]:


import re


# In[85]:


corpus=[]
for i in range(len(sentences)):
    review=re.sub('[^a-zA-Z]',' ',sentences[i])
    review=review.lower()
    corpus.append(review)


# In[86]:


corpus


# In[93]:


#stemming
for i in corpus:
    for word in i:
        print(stemmer.stem(word))
    


# In[94]:


stopwords.words('english')


# In[95]:


for i in corpus:
    words=nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(lemmatizer.lemmatize(word))


# In[104]:


#bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[105]:


X=cv.fit_transform(corpus)


# In[107]:


cv.vocabulary_


# In[108]:


corpus[0]


# In[109]:


X[0].toarray()


# # #day-3

# In[122]:


#TF_IDF
from sklearn.feature_extraction.text import TfidfVectorizer

cv=TfidfVectorizer(ngram_range=(3,3),max_features=10)
X=cv.fit_transform(corpus)


# In[123]:


corpus[0]


# In[124]:


X[0].toarray()


# # word to vec 

# In[125]:


get_ipython().system(' pip install gensim')


# In[127]:


import gensim


# In[128]:


from gensim.models import Word2Vec, KeyedVectors


# In[130]:


import gensim.downloader as api
wv=api.load('word2vec-google-news-300')
vec_king=wv['king']


# In[131]:


vec_king


# In[140]:


wv.most_similar('man')


# In[141]:


wv.most_similar('king')


# In[142]:


wv.similarity('man','king')


# In[143]:


wv.similarity('html','programer')


# # #project day

# In[1]:


import pandas as pd


# In[8]:


messages = pd.read_csv("C:/Users/ARPITA SINGH/Desktop/SMSSpamCollection.txt", sep='\t',
                           names=["label", "message"])


# In[9]:


messages


# In[10]:


messages['message'].loc[451]


# In[12]:


messages.shape


# In[17]:


#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')


# In[18]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[19]:


corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[20]:


corpus


# In[35]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500,binary=True)
X = cv.fit_transform(corpus).toarray()


# In[37]:


X
X.shape


# In[38]:


y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# In[39]:


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[40]:


X_train


# In[41]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)


# In[42]:


#prediction
y_pred=spam_detect_model.predict(X_test)


# In[43]:


from sklearn.metrics import accuracy_score,classification_report


# In[44]:


score=accuracy_score(y_test,y_pred)
print(score)


# In[45]:


from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))


# In[29]:


# Creating the TFIDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features=2500)
X = tv.fit_transform(corpus).toarray()


# In[30]:


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[31]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)


# In[32]:


#prediction
y_pred=spam_detect_model.predict(X_test)


# In[33]:


score=accuracy_score(y_test,y_pred)
print(score)


# In[34]:


from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))


# # Word2vec Implementation

# In[46]:


get_ipython().system('pip install gensim')


# In[47]:


import gensim.downloader as api

wv = api.load('word2vec-google-news-300')


# In[ ]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


# In[ ]:


corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[48]:


from nltk import sent_tokenize
from gensim.utils import simple_preprocess


# In[49]:


corpus[0]


# In[50]:


words=[]
for sent in corpus:
    sent_token=sent_tokenize(sent)
    for sent in sent_token:
        words.append(simple_preprocess(sent))


# In[51]:


words


# In[54]:


import gensim


# In[55]:


### Lets train Word2vec from scratch
model=gensim.models.Word2Vec(words,window=5,min_count=2)


# In[56]:


model.wv.index_to_key


# In[57]:


model.corpus_count


# In[58]:


model.epochs


# In[59]:


model.wv.similar_by_word('kid')


# In[60]:


model.wv['kid'].shape


# In[61]:


def avg_word2vec(doc):
    # remove out-of-vocabulary words
    #sent = [word for word in doc if word in model.wv.index_to_key]
    #print(sent)
    
    return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key],axis=0)
                #or [np.zeros(len(model.wv.index_to_key))], axis=0)


# In[62]:


get_ipython().system('pip install tqdm')


# In[63]:


from tqdm import tqdm


# In[64]:


words[73]


# In[65]:


type(model.wv.index_to_key)


# In[68]:


#apply for the entire sentences
import numpy as np
X=[]
for i in tqdm(range(len(words))):
    print("Hello",i)
    X.append(avg_word2vec(words[i]))


# In[69]:


type(X)


# In[70]:


X_new=np.array(X)


# In[71]:


X_new[3]


# In[72]:


X_new.shape


# In[ ]:




