import datetime
import numpy as np
import tensorflow as tf


class SmartDict(dict):
    def __init__(self, *args, **kwargs):
        super(SmartDict, self).__init__(*args, **kwargs)
        self['_prefix'] = None

    def set_prefix(self, prefix):
        self['_prefix'] = prefix

    def clear_prefix(self):
        self['_prefix'] = None

    def _get_name(self, key):
        if self['_prefix']:
            return '%s_%s' % (self['_prefix'], key)
        return key

    def __setattr__(self, key, value):
        self[self._get_name(key)] = value

    def __getattr__(self, key):
        if key == '_prefix':
            return self[key]
        if self._get_name(key) in self:
            return self[self._get_name(key)]
        raise AttributeError()


class Dataset(SmartDict):
    """docstring for Dataset"""
    def __init__(self, paths=None):
        super(Dataset, self).__init__()
        if paths:
            self.load((np.load(paths.x), np.load(paths.y)))

    def load(self, data):
        self.x, self.y = data
        self.num_examples = self.x.shape[0]
        
    def split_off(self, size):
        self.permute()
        new_dataset = Dataset()
        new_dataset.load((self.x[-size:], self.y[-size:]))
        self.load((self.x[:-size], self.y[:-size]))
        return new_dataset

    def random_swaps(self):
        for i, review in enumerate(self.x):
            dots = np.where(review == 2)[0]
            num_sents = dots.shape[0]
            if num_sents:
                sents = np.split(review, dots)
                perm = np.arange(num_sents)
                a, b = np.random.choice(num_sents, 2)
                sents[a], sents[b] = sents[b], sents[a]
                self.x[i] = np.concatenate(sents)

    def permute(self):
        perm = np.random.permutation(self.num_examples)
        self.x = self.x[perm]
        self.y = self.y[perm]

    def get(self, name, size):
        return self[name][:size]

    def map(self, wordmap):
        for i, review in enumerate(self.x):
            for j, word in enumerate(review):
                self.x[i][j] = wordmap[self.x[i][j]]


def compress_embeddings(data, embeddings):
    compressed_embeddings = []
    wordmap = dict()
    for name, dataset in data.items():
        for review in dataset.x:
            for wordid in review:
                if wordid not in wordmap:
                    wordmap[wordid] = len(wordmap)
                    compressed_embeddings.append(embeddings[wordid])

    for name, dataset in data.items():
        dataset.map(wordmap)

    return np.array(compressed_embeddings), len(compressed_embeddings)

def create_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def move_average(average, new_value):
    return average * 0.9 + 0.1 * new_value

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1)[:,None])
    return e_x / e_x.sum(axis=1)[:,None]

def get_date():
    return datetime.datetime.now().strftime('%02m%02d_%02H%02M%02S')

def get_model_name(params):
    return 'lstm_%s' % get_date()

def describe_params(params):                
    print('-'*30)
    for k, v in sorted(params.values().items()):
        print('%s: %s' % (k, v))