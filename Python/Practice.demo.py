# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
#
# mystr = "一枝花·咏喜雨::张养浩::用尽我为国为民心，(祈下些值金值玉雨)，数年空盼望，一旦遂沾濡，唤省焦枯，' \
#         '喜万象春如故，恨流民尚在途，留不住都弃业抛家，当不的也离乡背土。[梁州]恨不得把野草翻腾做菽粟，' \
#         '澄河沙都变化做金珠。直使千门万户()家豪(富，我)也不枉了受天禄。眼觑着灾伤教我没是处，只落得雪满头颅。' \
#         '[尾声]青天多谢相扶助，赤子从今罢叹吁。只愿得三日霖霪不停住，便下当街上似五湖，都渰了九衢，' \
#         '犹自洗[不尽从前受过的苦。]"
# # while True:
# #     if mystr.find('(') !=-1:
# #         first_in, mystr = mystr.split ("(",1)
# #         _, end_in = mystr.split(')',1)
# #         mystr = first_in + end_in
# #     else:
# #         break
# # while True:
# #     if mystr.find('[') != -1:
# #         first_in ,mystr = mystr.split('[',1)
# #         _,end_in = mystr.split(']',1)
# #         mystr = first_in + end_in
# #     else:
# #         break
#
# # from gensim.models import Word2Vec
# # from sklearn.decomposition import PCA
# # from matplotlib import pyplot
# #
# # sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
# # 			['this', 'is', 'the', 'second', 'sentence'],
# # 			['yet', 'another', 'sentence'],
# # 			['one', 'more', 'sentence'],
# # 			['and', 'the', 'final', 'sentence']]
# #
# #
# # model = Word2Vec(sentences,window=5, min_count=1)
# #
# # # 基于2d PCA拟合数据
# # X = model[model.wv.vocab]
# # print(X)
# # pca = PCA(n_components=2)
# # result = pca.fit_transform(X)
# #
# # pyplot.scatter(result[:,0] , result[:,1])
# # words = list(model.wv.vocab)
# # for i,word in enumerate(words):
# #     pyplot.annotate(word,xy=(result[i,0],result[i,1]))
# # pyplot.show()
# Themeword = [['' for i in range (5)] for i in range (15)]
# Themeword[0] = ['建功立业', '壮志', '报国', '厌战', '苦寒', '渴望', '和平', '琵琶', '胡琴', '边塞', '干戈', '玉门关', '楼兰', '天山', '征战']
# Themeword[1] = ['梅', '竹', '菊', '石', '柳', '松', '高尚', '纯洁', '乐观', '坚强', '高洁', '德馨', '坚韧', '磊落', '正直']
# Themeword[2] = ['山', '水', '田园', '隐逸', '世俗', '自然', '闲适', '高洁', '理想', '追求', '苦闷', '热爱', '恬静', '生活', '哲理']
# Themeword[3] = ['漂泊', '仕途', '作客', '离乱', '家书', '孤独', '难眠', '远望', '思乡', '怀念', '大雁', '秋思', '羁旅', '秋风', '红豆']
# Themeword[4] = ['感慨', '兴衰', '古今', '英雄', '建功立业', '物是人非', '古迹', '兴亡', '商女', '怀古', '宫', '亭', '寺庙', '赤壁', '盛衰']
# # print(Themeword[4][14])
# # a=[1,3,4,6]
# # print(max(a))
# # print(a.index(6))
#
# line = "我是谁我可以可以可以水诗呀"
# # for  i in range(1,8):
# print(line[0:1])
# # line.index(i,len(line)).count(line[i])
#     for j in line:
#         k = line.count(j)
#         if k > count[i]:
#             count[i] = k
#             value[i] = j
from gensim.models import word2vec


def Judge_Twowords_Yayun( word1, word2):
    Yayun = word1
    f = open("E:\Desk\MyProjects\Python/NLP_Demo1/Yayun_words.txt",'r',encoding='utf-8')
    array =[]
    while True:
        line = f.readline()
        if not line:
            break;
        line = str(line)
        array.append(line.replace(" ","").replace("	",""))
    f.close()

    for i in range(13):
        if Yayun in array[i]:
            index = i
            break
    flag = False
    if word2 in array[index]:
        flag = True
    # print(index)
    # print(array[index])
    return flag

def Judge_word_in_Yayuntable(word):
    with open ("E:\Desk\MyProjects\Python/NLP_Demo1/Yayun_words.txt", 'r', encoding='utf-8') as f:
        content = f.read()
    if word in content:
        return True
    else:
        return False

from SQL_operator import SunckSql
from flask import Flask,request

app = Flask(__name__)

ipv4 = "127.0.0.1"  # 局域网IP地址
s = SunckSql(ipv4 , 'root','123456','practice')
@app.route("/" , methods=['GET','POST'])  # /服务器可设置多个路径，提供访问
def hello():
    print(request.args) #ImmutableMultiDict([('userName', '小悦悦')])

    # 127.0.0.1:8080/?userName=小悦悦
    name = request.args['userName'] # 获取 json 值
    print(name)  # 小悦悦
    sql = "Select * from userinforma WHERE userName = '"+name+"'"
    result = s.get_one(sql)  # ('2', '2', '小悦悦', '女', '19960564')
    print(result)
    return 'hello word'

if __name__ == "__main__":
    # word1 = "心"
    # word2 = "端"
    # t = Judge_Twowords_Yayun(word1 ,word2 )
    # tt = Judge_word_in_Yayuntable(word1)
    # print(t)
    # app.run(host=ipv4, port=8080)

    path_save = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\wiki_model\wiki_corpus.model"
    # 加载训练好的模型wiki_corpus.model
    model2 = word2vec.Word2Vec.load (path_save)
    word1='边塞'
    word2='大漠'
    word3='梅花'
    similar = model2.similarity(word1,word2)
    similar2=model2.similarity(word1,word3)
    print(similar)
    print(similar2)


