
from config import *
import data
import model


def file_id( filename):
    """pretreatment"""
    poems = []
    file = open (filename, "r", encoding='utf-8')
    for line in file:  # every line is a poem
        title, author, poem = line.strip ().split ("::")  # get title and poem
        poem = poem.replace (' ', '')  # 去空格
        if len (poem) < 10 or len (poem) > 512:  # 去 过长过短诗
            continue
        if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
            continue
        poem = '[' + poem + ']'  # add start and end signs
        poems.append (poem)

    # counting words
    # wordFreq Counter type, {'字'，字频 。。。} 从 高-->低
    wordFreq = collections.Counter ()
    for poem in poems:
        wordFreq.update (poem)
    wordFreq[" "] = -1

    wordPairs = sorted (wordFreq.items (), key=lambda x: -x[1])
    words, freq = zip (*wordPairs)
    wordNum = len(words)  # 整个诗歌训练集中字的总个数
    # print(self.words)
    # print(freq)
    # print(self.wordNum)
    # wordToID {'字'，'字频' 。。。} 按字频从高-->低
    wordToID = dict (zip (words, range(wordNum)))
    return wordToID

if __name__ == "__main__":
    # path = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\generate_poem\Poetry_class/biansai.txt"
    # w_to_id = file_id(path)
    # s= '单车欲问边，属国过居延。征蓬出汉塞，归雁入胡天。大漠孤烟直，长河落日圆。萧关逢候骑，都护在燕然。'
    # id =[]
    ss = '北风卷地白草折，胡天八月即飞雪。忽如一夜春风来，千树万树梨花开。散入珠帘湿罗幕，狐裘不暖锦衾薄。将军角弓不得控，都护铁衣冷难着。瀚海阑干百丈冰，愁云惨淡万里凝。中军置酒饮归客，胡琴琵琶与羌笛。纷纷暮雪下辕门，风掣红旗冻不翻。轮台东门送君去，去时雪满天山路。山回路转不见君，雪上空留马行处。'
    s = '北风卷地白草折,'
    print(len(s))
    # for i in range(len(s)):
    #     value = w_to_id[s[i]]
    #     id.append(value)
    # print(id)
