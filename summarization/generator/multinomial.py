# -*- coding: utf-8 -*-
__project__ = 'textsummary'
__author__ = 'Alfio Ferrara'
__email__ = 'alfio.ferrara@unimi.it'
__institution__ = 'Università degli Studi di Milano'
__date__ = '19 giu 2017'
__comment__ = '''
    Classes for generating summaries by multinomial language models
    '''
import numpy as np
import string
import nltk
import PyINSECT.NGramGraphCollector as n_gram
import pickle


class Supervised:

    def __init__(self):
        self._theta = None
        self.t_docs, self.models = {}, {}
        self.n_gram_graph = n_gram.NGramGraphCollector()

    def save(self, file_path):
        pickle.dump(self, open(file_path, "wb"))

    @property
    def topics(self):
        return [x for x in sorted(self.t_docs.keys())]

    def feed(self, topic, text, tokenizer):
        """
        Update size of topics and the corresponding model
        :param topic: a suitable topic name
        :param text: pure document text
        :param tokenizer: a class instance implementing the tokenize method (e.g., NLTK)
        :return:
        """
        text = "".join([t for t in text if t not in string.punctuation])
        self.n_gram_graph.addText(text)
        tokens = tokenizer.tokenize(text)
        try:
            self.t_docs[topic].append(tokens)
        except KeyError:
            self.t_docs[topic] = [tokens]

    def fit(self):
        """
        Generates the distribution theta according to
        the size of each topic. Creates also all the
        multinomial models for topics
        :return:
        """
        lns = np.array([len(self.t_docs[x]) for x in self.topics])
        self._theta = np.random.dirichlet(lns)
        for topic, doc in self.t_docs.items():
            words = [y for x in x for x in doc for y in x]
            unigram_probs = np.zeros(len(words))
            unigram = nltk.FreqDist(words)
            for i, w in enumerate(words):
                unigram_probs[i] = float(unigram[w]) / len(words)
            unigram_probs = np.random.dirichlet(unigram_probs)
            cfreq_2gram = nltk.ConditionalFreqDist(nltk.bigrams(words))
            bigram = nltk.ConditionalProbDist(cfreq_2gram, nltk.MLEProbDist)
            self.models[topic] = (words, unigram_probs, bigram)

    def sample_len(self, percentage=100):
        """
        Provides a random length according to the dataset.
        :param percentage: Determines the percentage of text len to take as input
        :return:
        """
        lns = np.array([int(percentage / 100.0 * len(y)) for x in self.topics for y in self.t_docs[x]])
        l_dist = np.random.dirichlet(lns)
        return np.random.choice(lns, size=1, p=l_dist)[0]

    def generate(self, size=100, n_gram_validation=False,
                 percentage=True, single_topic=False,
                 n_gram_trials=100, first_word=None):
        """
        Generates a text of length ln by picking up a topic according to
        theta and a word according to topic probability
        :param n_gram_trials:
        :param n_gram_validation: if True uses nGram to evaluate the probability to keep the text as is
        if not, a new word is extracted for n_gram_trials times
        :param single_topic: if True all the words are generated according to the same topic
        :param size: If percentage=False, it is the actual size of generated text.
        Otherwise, len is generated randomly to be a fraction of the dataset text len
        :param percentage:
        :return:
        """
        text, txt_topics, last_word, sigmas = [], [], None, []
        if percentage:
            ln = self.sample_len(percentage=size)
        else:
            ln = size
        t = np.random.choice(self.topics, size=1, p=self._theta)[0]
        for iteration in range(0, ln):
            print 'generating word', iteration, 'of', ln
            if iteration > 0 and not single_topic:
                t = np.random.choice(self.topics, size=1, p=self._theta)[0]
            txt_topics.append(t)
            if iteration == 0:
                w = np.random.choice(self.models[t][0], size=1, p=self.models[t][1])[0]
                text.append(w)
            else:
                try:
                    w = self.models[t][2][last_word].generate()
                except IndexError:
                    w = np.random.choice(self.models[t][0], size=1, p=self.models[t][1])[0]
                if n_gram_validation:
                    c, sm = w, 0.0
                    for trial in range(0, n_gram_trials):
                        candidate_text = " ".join(text + [w])
                        p = self.n_gram_graph.getAppropriateness(candidate_text)
                        coin = np.random.uniform()
                        if coin <= p:
                            c = w
                            sm = p
                            break
                    text.append(c)
                    sigmas.append(sm)
                else:
                    text.append(w)
                    sigmas.append(self.n_gram_graph.getAppropriateness(" ".join(text)))
            last_word = w
        return text, txt_topics, sigmas

    @staticmethod
    def _unigram_prob(word, f):
        return float(f[word]) / sum(f.values())
