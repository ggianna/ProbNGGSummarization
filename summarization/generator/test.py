# -*- coding: utf-8 -*-
__project__ = 'textsummary'
__author__ = 'Alfio Ferrara'
__email__ = 'alfio.ferrara@unimi.it'
__institution__ = 'Università degli Studi di Milano'
__date__ = '19 giu 2017'
__comment__ = '''
    Simple playground for multinomial generators
    '''
from multinomial import Supervised
import os, codecs
from nltk.tokenize import WhitespaceTokenizer
import pickle


S = Supervised()
tokenizer = WhitespaceTokenizer()
docs = '/Users/alfio/Research/NCSR/textsummary/SourceTextsV2b/english'
model_dir = '/Users/alfio/Programming/Python2.7/textsummary/data'

feed = False

if feed:
    # FEED
    file_list = os.listdir(docs)
    for i, f in enumerate(file_list):
        topic = "".join(f[:4])
        fp = os.sep.join([docs, f])
        print 'feeding', i+1, 'of', len(file_list), f
        with codecs.open(fp, mode='rU', encoding='utf-8') as of:
            S.feed(topic, of.read(), tokenizer)

    # FIT
    print 'Fit model'
    S.fit()
    S.save(os.sep.join([model_dir, 'multinomial.ngg']))
else:
    f = open(os.sep.join([model_dir, 'multinomial.ngg']), 'rb')
    S = pickle.loads(f.read())

# GENERATE
for i in range(0, 1):
    text, topics, sigmas = S.generate(size=2,
                                      n_gram_validation=True,
                                      n_gram_trials=10,
                                      percentage=True,
                                      single_topic=True)
    print " ".join(text)
    print topics
    print sigmas
    print ""
    text, topics, sigmas = S.generate(size=2,
                                      n_gram_validation=False,
                                      n_gram_trials=10,
                                      percentage=True,
                                      single_topic=True)
    print " ".join(text)
    print topics
    print sigmas
