"""
    word2vec 模型
    1.根据用户输入关键词判断其所属类别，用wiki_model(以维基百科为语料得到预测模型)
      本来想着用全唐诗作为语料，但实践发现，若用户输入的关键词不在全唐诗中（ep:壮志难酬）
      则模型无法实施预测，根本原因语料数据太小，无法加载大量词汇
    2.根据判断后的所属类别，分别生成5类诗的 ***.model ,以所属类别的 .model来预测与用户关键词
      最相近的 6 个词，去重后用作诗中每句的备选开头字
"""
import re
import jieba
from gensim.models import word2vec
def file_deal(path_before,path_after):
        # 处理停用词
        stop_word = []
        with open('E:\Desk\MyProjects\Python/NLP_Demo1\CNstopwords.txt','r',encoding='utf-8') as f_reader:
            for line in f_reader:
                line = str(line.replace('\n','').replace('\r','').split())
                stop_word.append(line)
        stop_word = set(stop_word)

        rules = u'[\u4e00-\u9fa5]+'
        pattern = re.compile(rules)
        f_write = open(path_after,'a+',encoding='utf-8')
        f_reader2 = open(path_before , 'r',encoding='utf-8')
        while True:
            line = f_reader2.readline()
            if not line:
                break
            title,author,poem = line.strip().split("::")
            poem = poem.replace('\n','').replace('\r','').split()
            poem = str(poem)
            poem = " ".join(jieba.cut(poem))
            seg_list = pattern.findall(poem)

            word_list = []
            for word in seg_list:
                if word not in stop_word:
                    word_list.append(word)
            line = " ".join(word_list)
            f_write.write(line + '\n')
            f_write.flush()
        f_write.close()

    #训练模型

def practice_model(path_6shi_deal,path_save_key):
        # path 全唐诗路径(处理后的) ，path_save_key 生成模型后保存的路径
        path = path_6shi_deal
        sentences = word2vec.Text8Corpus(path) #加载文件
        """"
        sentences 分析的预料，可以是一个列表，或从文件中遍历读出
        size ：词向量的维度， 默认值 100 ， 语料> 100M, size值应增大
        window: 词向量上下文最大距离，默认 5，小语料值取小
        sg: 0--> CBOW模型    1--> Skip-Gram模型
        hs: 0 负采样 ， 1 Hierarchical Softmax
        negative ： 使用负采样 的个数， 默认 5
        cbow_mean:  CBOW中做投影，默认 1，按词向量平均值来描述
        min_count: 计算词向量的最小词频 ，可以去掉一些生僻词（低频），默认 5
        iter:随机梯度下降法中迭代的最大次数，默认 5，大语料可增大
        alpha: 迭代的初始步长，默认 0.025
        min_alpha:最小迭代步长，随机梯度下降法，迭代中步长逐步减小
        """
        # 调参 iter 训练轮次    size 大小
        # 全唐诗模型性 iter = 20 ，size = 300
        # 其余5类是模型  iter = 20 ， size = 200
        model = word2vec.Word2Vec(sentences,window=4, iter=10,alpha =0.025,min_count=4, size=100)
        # 保存模型   path_save模型路径
        path_save = path_save_key
        model.save(path_save)




if __name__ == '__main__':
    path_6shi_deal = "E:\Desk\MyProjects\Python\Yuliaodeal/sixiang.txt"
    path_save_key = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir\Keyword_sixiang.model'
    # practice_model (path_6shi_deal, path_save_key)
    # 某一类中，返 6个相似词， 关键词一定要在全唐诗中
    key_word = '秋思'
    model = word2vec.Word2Vec.load(path_save_key)

    words = model.most_similar (positive=[key_word], topn=6)
    for i in range(6):
        print(words[i])


