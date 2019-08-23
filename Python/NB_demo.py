"""
    朴素贝叶斯分类模型
    用于全唐诗分类，但实践效果不好，最终没用采用
"""
from config import*
class NB():
    global stopWords
    global  classtags_list,train_data
    # 读取停用词
    stopWords = open ("E:/Desk\MyProjects/Python/NLP_Demo1/CNstopwords.txt", encoding='UTF-8').read ().split ("\n")
    # 以空格来分词，对原始数据进行预处理
    def preprocess_only(path):
        text_with_space = ''
        testarray = []
        file = open(path, 'r', encoding='utf-8',errors='ignore')
        while True:
            textfile = file.readline()
            if not textfile:
                break
            # textcute = jieba.cut(textfile)
            # print(textcute)
            inputData = "".join (re.findall(u'[\u4e00 -\u9fa5]+', textfile))
            wordList = "/".join(jieba.cut(inputData))
            listOfTokens = wordList.split("/")
        # return [tok for tok in listOfTokens if (tok not in stopWords and len(tok) >= 1)]
            for word in listOfTokens:
                text_with_space += word + ' '  # 以空格划分
            testarray.append(text_with_space)
            # print(testarray)
        return testarray


    # 训练---以每首诗来进行分类，设置标签
    def preprocess(path,class_title):
        text_with_space = ""
        class_tt = []
        textarray = []
        file = open(path,'r',encoding='utf-8',errors='ignore')
        while True:
            mystr = file.readline()
            if not mystr:
                break
            # textcute = jieba.cut(mystr) # 分词
            # textcute = textParse(mystr)
            # print(textcute)
            inputData = "".join (re.findall (u'[\u4e00 -\u9fa5]+', mystr))
            wordList = "/".join (jieba.cut (inputData))
            listOfTokens = wordList.split ("/")
            for word in listOfTokens:
                text_with_space += word + " " # 用空格划分
            textarray.append(text_with_space)
            class_tt.append(class_title)

        return textarray, class_tt

    # 每类诗的标签，各类诗，统一整合成train_data训练数据集，根据标签进行有监督训练
    def deal_Data(self):
        processed_textdata1,class1 = NB.preprocess("E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar/jingwu.txt",'写景咏物')
        processed_textdata2,class2 = NB.preprocess("E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar/yongshi.txt", '咏史怀古')
        processed_textdata3,class3= NB.preprocess("E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar/shanshui.txt",'山水田园')
        processed_textdata4,class4= NB.preprocess("E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar/biansai.txt",'边塞征战')
        processed_textdata5,class5 = NB.preprocess("E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar/sixiang.txt",'羁旅思乡')

        train_data = processed_textdata1 + processed_textdata2 + processed_textdata3 + processed_textdata4 + processed_textdata5
        classtags_list= class1 + class2 + class3 + class4 + class5

        print(str(len(classtags_list)) + '--' + str(len(train_data)))
        return train_data,classtags_list

    # 生成训练模型，对测试数据进行预测
    # 训练数据格式：   题目::作者::诗内容
    def NBpredict(self):
        train_data,classtags_list=NB.deal_Data(self)
        """
            CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵
            get_feature_names()可看到所有文本的关键字
            vocabulary_可看到所有文本的关键字和其位置
            toarray()可看到词频矩阵的结果
        """
        count_vector = CountVectorizer()
        vecot_matrix = count_vector.fit_transform(train_data) # 将词语转换成词频矩阵
        print(vecot_matrix)
        """
        TfidfTransformer是统计CountVectorizer中每个词语的tf-idf权值
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        vectorizer.fit_transform(corpus)将文本corpus输入，得到词频矩阵
        将这个矩阵作为输入，用transformer.fit_transform(词频矩阵)得到TF-IDF权重矩阵
        TfidfTransformer + CountVectorizer  =  TfidfVectorizer  
        这个成员的意义是词典索引，对应的是TF-IDF权重矩阵的列，只不过一个是私有成员，一个是外部输入，原则上应该保持一致。
            use_idf：boolean， optional 启动inverse-document-frequency重新计算权重
        """
        train_tfidf = TfidfTransformer(use_idf=False).fit_transform(vecot_matrix)
        clf = MultinomialNB().fit(train_tfidf ,classtags_list)
        # 测试数据
        path_name= 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/test_jar/test_bert'
        testset = NB.preprocess_only(path_name)
        print(len(testset))
        for i in range(0,len(testset)):
            new_count_vector = count_vector.transform([testset[i]])
            # 用 transformer.fit_transform(词频矩阵） 得到 TF-IDF 权重矩阵
            new_tfidf = TfidfTransformer(use_idf = False).fit_transform(new_count_vector)
            print(new_tfidf)
            # 根据 由训练集而得到的分类模型，clf ,由 测试集的 TF-IDF权重矩阵来进行预测分类
            predict_result = clf.predict(new_tfidf)
            print(predict_result)
            # return train_data,classtags_list

if __name__ == '__main__':
    # preprocess ( 'E:/NLPfile/zz_test.txt', '边塞')
    nb =  NB()
    nb.NBpredict()
    # preprocess_only("E:/Desk/MyProjects/Python/NLP_Demo1/File_jar/test_jar/test_bert")