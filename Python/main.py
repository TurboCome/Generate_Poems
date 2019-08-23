""""
    主函数
    1.根据用户输入的根据词，判断所属类别
    2.在指定类别中，根据关键词生成备选词（字）
    3.LSTM模型进行预测
"""
from config import *
import data
import model
import word2vec_demo


class Main():
    def Finalsummary(Num,JueLv,Keyword):
        # characters = input("Please input characters：")
        characters = Keyword
        # # 根据关键词返回类别标签
        label = word2vec_demo.Word2vec_similar.class_tags(characters)
        print(label)
        Imbalance_words = word2vec_demo.Word2vec_similar.similar_6words(characters, label)
        if  '边塞征战'== label:
            class_tag = 'biansai'
        elif '写景咏物'== label:
            class_tag = 'jingwu'
        elif '山水田园'== label:
            class_tag = 'shanshui'
        elif '思乡羁旅'== label:
            class_tag = 'sixiang'
        else:
            class_tag = 'poetrySong'

        checkpointsPath = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\generate_poem/" + class_tag    # checkpoints location
        trainPoems = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\generate_poem\Poetry_class/" + class_tag+ ".txt"  # training file location
        # 训练数据时用，依次更改诗的种类，路径
        # trainPoems = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\generate_poem\Poetry_class/yongshi.txt"
        # checkpointsPath = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\generate_poem/yongshi"
        trainData = data.POEMS(trainPoems)
        MCPangHu = model.MODEL(trainData)  # 带参初始化
        #***** 分别训练5类模型
        # MCPangHu.train(checkpointsPath)
        poems = MCPangHu.testHead(characters,Imbalance_words,checkpointsPath,Num,JueLv)
        return poems

if __name__ == "__main__":
    characters = "田园"  # 用户输入关键词
    Num = 7
    JueLv = 8
    l  = Main.Finalsummary(Num , JueLv, characters)
    print(l)