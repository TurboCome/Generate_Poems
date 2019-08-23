import tensorflow as tf
import numpy as np
import argparse
import os
import io
import random
import time
import collections
import re
import operator
import jieba
import fastText

# 爬虫所需工具包
import sys
from urllib import request
from bs4 import BeautifulSoup
from lxml import html
# word2vec 所需工具包
from gensim.models import word2vec
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
from matplotlib import pyplot
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# TF-IDF所需工具包
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 云图展示
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from os import path


# 设置全局变量参数 + LSTM 模型参数
batchSize = 32
learningRateBase = 0.001     # 学习率
learningRateDecayStep = 1000 # 每隔 1000 步学习率下降
learningRateDecayRate = 0.95
epochNum = 30                    # train epoch
generateNum = 1                  # number of generated poems per time


Yayun = ""   # 第二句最后一个字， （押韵字）
Yayunlist=[]
saveStep = 1000                  # save model every savestep
type = "poetryTang"                   # dataset to use, shijing, songci, etc



