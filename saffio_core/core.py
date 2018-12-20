import numpy as np
import pandas as pd
import re

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import stopwords as stt
from nltk.corpus import stopwords as ste
from nltk.stem import WordNetLemmatizer
stopwords_t = stt.words('thai')
stopwords_e = ste.words('english')
lemmer = WordNetLemmatizer()


def to_one_hot_vector(c, dic):
    """Obsolete: Use embedding feature in the tensor library instead."""
    res = np.zeros(len(dic))
    if c in dic:
        res[dic[c]] = 1
    return res


def to_vector(tok_ls, idx):
    res = np.zeros(len(idx))
    for t in tok_ls:
        if t in idx:
            res[idx[t]] = 1
    return res


def padd_to_x(arr, x):
    res = np.zeros(x)
    res[:arr.shape[0]] = arr
    return res


def to_hotbox(se, dic, max_len=None):
    """Note: only useful for RNN"""
    hot_lst = [np.reshape([to_one_hot_vector(c, dic) for c in x], -1) for x in se]
    if max_len is None:
        max_len = np.max([len(x) for x in hot_lst])
    else:
        max_len = max_len * len(dic)
    hot_lst = [padd_to_x(x, max_len) for x in hot_lst]
    hot_lst = np.reshape(hot_lst, [len(se), max_len // len(dic), len(dic)])
    return hot_lst


def to_hotbox_reverse(se, dic):
    """Note: only useful for RNN"""
    hot_lst = [np.reshape([to_one_hot_vector(c, dic) for c in x[::-1]], -1) for x in se]
    max_len = np.max([len(x) for x in hot_lst])
    hot_lst = [padd_to_x(x, max_len) for x in hot_lst]
    hot_lst = np.reshape(hot_lst, [len(se), max_len // len(dic), len(dic)])
    return hot_lst


def to_dist(dic):
    """Note: dict value have to be a number"""
    dic_dist = {}
    sum_val = sum(dic.values())
    for c in dic:
        dic_dist[c] = dic[c] / sum_val
    return dic_dist


def get_char_hist_dic(se):
    dic = {}
    for x in se:
        for c in x:
            if c in dic:
                dic[c] += 1
            else:
                dic[c] = 1
    return to_dist(dic)


def get_hist_dic(se):
    """Obsolete: use get_char_hist_dic instead"""
    dic = {}
    for x in se:
        for c in x:
            if c in dic:
                dic[c] += 1
            else:
                dic[c] = 1
    return to_dist(dic)


def get_dict_high_percent(dic, p):
    ret = {}
    acc = 0
    for key, value in sorted(dic.items(), key=lambda k: (k[1], k[0]), reverse=True):
        acc += value
        ret[key] = value
        if acc >= p:
            return ret
    return ret


def index_mapping(ls):
    idx = {}
    rev_idx = {}
    for i in range(len(ls)):
        idx[ls[i]] = i
        rev_idx[i] = ls[i]
    return idx, rev_idx


def get_char_dict(se):
    hist = pd.DataFrame(sorted(get_hist_dic(se).items(), key=lambda x: x[1], reverse=True))
    hist['index'] = hist.index
    char_dict = hist.set_index(0).to_dict()['index']
    char_dict_rev = hist.to_dict()[0]
    return char_dict, char_dict_rev


def str_cleaner(s):
    """Will only leave Thai, English and numbers. Everything else will be filtered out"""
    if s is None:
        return ''
    upp = s.upper()
    clean_str = re.sub('[^ก-ฮะ-ูเ-์A-Z0-9]', ' ', upp)
    clean_str = re.sub('\s+', ' ', clean_str).strip()
    return clean_str


def toknice(st):
    """Note: Use for bag of words model"""
    return [lemmer.lemmatize(s) for s in word_tokenize(st)]


def get_tok_dic(se):
    """Note: Use for bag of words model"""
    di = {}
    for s in se:
        if s is None:
            continue
        ar = toknice(s)
        for a in ar:
            if a in di:
                di[a] += 1
            else:
                di[a] = 1
    for s in stopwords_t:
        if s in di:
            del di[s]
    for s in stopwords_e:
        s = str.upper(str_cleaner(s))
        if s in di:
            del di[s]
    if ' ' in di:
        del di[' ']
    return di


def prune(sls, rand_state):
    if len(sls) == 0:
        return sls
    tal = max(int(rand_state.rand() * len(sls)), 1)
    return rand_state.choice(sls, tal)


def to_skipgram(ls, neighbour=2):
	ret = []
	le = len(ls)
	if le == 1:
		return [[ls[0], 0]]
	for i in range(le):
		for j in range(1, neighbour+1):
			if i-j >= 0:
				ret.append([ls[i], ls[i-j]])
			if i+j < le:
				ret.append([ls[i], ls[i+j]])
	return ret
