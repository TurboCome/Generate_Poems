"""
    LSTM模型 写诗前的数据处理，返回模型的输入量 X
"""
from config import *

class POEMS():
    erase = []
    "poem class"
    def __init__(self,filename):
        """pretreatment"""
        poems = []
        file = open(filename, "r",encoding='utf-8')
        for line in file:  #every line is a poem
            title, author, poem = line.strip().split("::")  #get title and poem
            poem = poem.replace(' ','')   # 去空格
            if len(poem) < 10 or len(poem) > 512:  #去 过长过短诗
                continue
            if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
                continue
            poem = '[' + poem + ']' #add start and end signs
            poems.append(poem)

        #counting words
        # wordFreq Counter type, {'字'，字频 。。。} 从 高-->低
        wordFreq = collections.Counter()
        for poem in poems:
            wordFreq.update(poem)
        # print (wordFreq)   #
        # erase words 生僻字
        for key in wordFreq:
            if wordFreq[key] < 2:
                POEMS.erase.append(key)
        # for key in erase:
        #     del wordFreq[key]
        #     for i in range(len(poems)):
        #         str_key = poems[i]
        #         str_key = str(str_key)
        #         if key in str_key:
        #             str_key.strip(key)
        #         poems[i]=str_key

        wordFreq[" "] = -1

        wordPairs = sorted(wordFreq.items(), key=lambda x: -x[1])
        self.words, freq = zip(*wordPairs)
        self.wordNum = len(self.words)  # 整个诗歌训练集中字的总个数
        # print(self.words)
        # print(freq)
        # print(self.wordNum)
        # wordToID {'字'，'字频' 。。。} 按字频从高-->低
        self.wordToID = dict(zip(self.words, range(self.wordNum)))
        # poem to vector ,诗中每个字 对应的 id
        poemsVector = [([self.wordToID[word] for word in poem]) for poem in poems]
        # print(poemsVector)
        # 分析训练词向量，测试词向量
        self.trainVector = poemsVector
        self.testVector = []
        print("训练样本总数： %d" % len(self.trainVector))
        # print("测试样本总数： %d" % len(self.testVector))


    def generateBatch(self, isTrain=True):
        """
        每轮仅输入 batchSize 个样本数据，
        :param isTrain:
        :return:
        """
        if isTrain:
            poemsVector = self.trainVector  # 训练数据所对应的 id ，一维
        else:
            poemsVector = self.testVector
        # 将词向量打乱
        random.shuffle(poemsVector)
        # batchSize 同时运行的一批样本数量，较小时，有利于模型找到最优解，
        # 较大时，缩短训练时间， 与内存大小有关
        batchNum = (len(poemsVector) - 1) // batchSize
        # batchNum 表示 一个epoch下，所需计算的迭代次数，一次迭代可运算batchSize个样本
        X = []
        Y = []

        for i in range(batchNum):
            batch = poemsVector[i * batchSize : (i + 1) * batchSize]
            # 每个 batch 批次中的向量，都计算其长度，在这些长度中找到最长的那个
            maxLength = max([len(vector) for vector in batch])
            # 添加空格，将所有不足最长向量的vector（矩阵：batchSize ， maxLength） 补齐为最长向量
            # 定义全空格-id， 再赋值给定 诗-id，
            temp = np.full((batchSize, maxLength), self.wordToID[" "], np.int32)
            for j in range(batchSize):
                temp[j, :len(batch[j])] = batch[j]

            X.append(temp)
            temp2 = np.copy(temp)
            temp2[:, :-1] = temp[:, 1:] # Y 错开一个字，
            Y.append(temp2)
            print(X)
            print(Y)
        return X, Y

if __name__ == "__main__":
    path = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/test_jar\zz_test.txt"
    p = POEMS(path)
    print(p.trainVector)
    print(p.wordToID)
    print('\n')
    print(p.wordNum)
