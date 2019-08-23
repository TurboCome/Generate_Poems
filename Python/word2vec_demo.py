"""
    word2vec 模型
    1.根据用户输入关键词判断其所属类别，用wiki_model(以维基百科为语料得到预测模型)
      本来想着用全部诗歌数据集作为语料，但实践发现，若用户输入的关键词不在诗歌中（ep:壮志难酬）
      则模型无法实施预测，根本原因语料数据太小，无法加载大量词汇
    2.根据判断后的所属类别，分别生成5类诗的 ***.model ,以所属类别的 .model来预测与用户关键词
      最相近的 6 个词，去重后用作诗中每句的备选开头字
"""

from config import *


class Word2vec_similar ():
    def __init__(self):
        pass

    # 处理数据（若分类时已经处理过，则不用再次处理）
    def file_deal(path_before, path_after):
        # 处理停用词
        stop_word = []
        with open ('E:\Desk\MyProjects\Python/NLP_Demo1\CNstopwords.txt', 'r', encoding='utf-8') as f_reader:
            for line in f_reader:
                line = str (line.replace ('\n', '').replace ('\r', '').split ())
                stop_word.append (line)
        stop_word = set (stop_word)

        rules = u'[\u4e00-\u9fa5]+'
        pattern = re.compile (rules)
        f_write = open (path_after, 'a+', encoding='utf-8')
        f_reader2 = open (path_before, 'r', encoding='utf-8')
        while True:
            line = f_reader2.readline ()
            if not line:
                break
            title, author, poem = line.strip ().split ("::")
            poem = poem.replace ('\n', '').replace ('\r', '').split ()
            poem = str (poem)
            poem = " ".join (jieba.cut (poem))
            seg_list = pattern.findall (poem)

            word_list = []
            for word in seg_list:
                if word not in stop_word:
                    word_list.append (word)
            line = " ".join (word_list)
            f_write.write (line + '\n')
            f_write.flush ()
        f_write.close ()

    # 训练模型
    def practice_model(path_6shi_deal, path_save_key):
        # path 全唐诗路径(处理后的) ，path_save_key 生成模型后保存的路径
        path = path_6shi_deal
        sentences = word2vec.Text8Corpus (path)  # 加载文件
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
        model = word2vec.Word2Vec (sentences, iter=20, min_alpha=0.005, min_count=8, size=150)
        # 保存模型   path_save模型路径
        path_save = path_save_key
        model.save (path_save)

    # 返回 6 个相似词
    def similar_6words(key_word, label):
        path_save = ""
        if label == '边塞征战':
            path_save = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir\Keyword_biansai.model"
        elif label == '写景咏物':
            path_save = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir\Keyword_jingwu.model"
        elif label == '山水田园':
            path_save = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir\Keyword_shanshui.model"
        elif label == '思乡羁旅':
            path_save = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir\Keyword_sixiang.model'
        else :
            path_save = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir\Keyword_yongshi.model"
        # 维基百科 2 词 相似度计算
        # wiki_path_model="E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\wiki_model\wiki_corpus.model"
        # model = word2vec.Word2Vec.load(wiki_path_model)
        model = ""
        model = word2vec.Word2Vec.load(path_save)
        re_word = []
        # 异常处理，当语料中无此关键词时，先取词前2个字，若还没有，取第一个字
        try:
            model.most_similar (positive=[key_word], topn=6)
        except:
            key_word = key_word[0:2]
            try:
                model.most_similar (positive=[key_word], topn=6)
            except:
                key_word = key_word[0]
        print(key_word)
        similary_words = model.most_similar(positive=[key_word] , topn = 6)
        for e in similary_words:
            print(e[0], e[1])
            re_word.append(e[0])

        # 可视化展示
        # 根据 model中的词，来做与关键词-距离的图像， 基于2d PCA拟合数据
        # X = model[model.wv.vocab]
        # print (X)
        # pca = PCA(n_components=2)
        # result = pca.fit_transform (X)
        # # 加载中文 字体
        # font = FontProperties (fname=r"C:\Windows\Fonts\simsun.ttc", size=16)
        # pyplot.scatter (result[:, 0], result[:, 1])
        # words = list (model.wv.vocab)
        # for i, word in enumerate (words):
        #     pyplot.annotate(word, fontproperties = font,xy=(result[i, 0], result[i, 1]))
        # pyplot.show ()
        return re_word

    # 计算 2 词之间的相似度
    def class_tags(str1):
        # 以全唐诗为语料，计算2词相似度（不用，语料少）
        # path_save = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir\Quanshi_Judge_6class.model'

        # 5类诗的主题词，用于与用户输入的关键词进行相似度计算，判断类别--》5*15 二维矩阵
        Themeword = [['' for i in range (5)] for i in range (15)]
        Themeword[0] = ['建功立业', '壮志', '报国', '厌战', '苦寒', '渴望', '和平', '琵琶', '胡琴', '边塞', '干戈', '玉门关', '楼兰', '天山', '征战']
        Themeword[1] = ['梅', '竹', '菊', '石', '柳', '松', '高尚', '纯洁', '乐观', '坚强', '高洁', '德馨', '坚韧', '磊落', '正直']
        Themeword[2] = ['山', '水', '田园', '隐逸', '世俗', '自然', '闲适', '高洁', '理想', '追求', '淡泊', '热爱', '恬静', '生活', '哲理']
        Themeword[3] = ['漂泊', '仕途', '作客', '离乱', '家书', '孤独', '难眠', '远望', '思乡', '怀念', '大雁', '秋思', '羁旅', '秋风', '红豆']
        Themeword[4] = ['感慨', '兴衰', '古今', '英雄', '建功立业', '物是人非', '古迹', '兴亡', '商女', '怀古', '宫', '亭', '寺庙', '赤壁', '无奈']

        # 以维基百科为语料，计算2词相似度
        path_save = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\wiki_model\wiki_corpus.model"
        # 加载训练好的模型wiki_corpus.model
        model2 = word2vec.Word2Vec.load (path_save)

        sum_similar = [0 for i in range (5)]
        for i in range (5):
            print ('\n')
            for j in range (15):
                # 异常处理，解决 wiki_model中没用此关键词的情况
                # 异常处理，当语料中无此关键词时，先取词前2个字，若还没有，取第一个字
                flag = False
                try:
                    sim_value = model2.similarity (str1, Themeword[i][j])
                    flag = True
                except:
                    print ("Error")
                    str1 = str1[0:2]
                    try:
                        sim_value = model2.similarity (str1, Themeword[i][j])
                        flag = True
                    except:
                        str1 = str1[0]

                if flag == False:
                    sim_value = model2.similarity (str1, Themeword[i][j])

                sum_similar[i] += sim_value
                # print(sim_value)

        max_value = max (sum_similar)  # 选出最相似的那类
        similar_index = sum_similar.index (max_value)  # 确定那类的标签
        label_tags = ["边塞征战", "写景咏物", "山水田园", "思乡羁旅", "咏史怀古"]
        return label_tags[similar_index]

    # 按词（字）频 进行云图展示
    def Picture_file(text_words):
        # 定义绝对路径地址
        __file__ = r"E:\NLPfile\\"
        # 把路径地址字符串转换为文件路径
        d = path.dirname (__file__)
        backgroug_Image = np.array (Image.open (path.join (d, "cloud2.png")))
        wc = WordCloud (
            background_color='white',  # 背景颜色
            mask=backgroug_Image,  # 设置背景为指定图片
            # font_path="‪C:\Windows\Fonts\汉仪雪君体简.ttf",  # 设置字体
            font_path="‪C:\Windows\Fonts\simkai.ttf",
            max_words=2000,  # 最大字频数
            max_font_size=150,  # #最大号字体，如果不指定则为图像高度
            random_state=1,
            scale=3  # 像素
        )
        wc.generate (text_words)
        # 根据图片颜色重新上色
        image_clors = ImageColorGenerator (backgroug_Image)
        plt.imshow (wc.recolor (color_func=image_clors))
        wc.to_file ("E:/NLPfile/cloud_yongshi.png")  # 保存图片
        plt.axis ("off")  # 去除坐标轴
        plt.show ()

    # 选择《诗学含英》中的6个 与主题词最相似的标题词
    def label_offen_words(character):
        before_key = 6
        file = open ("E:\Desk\MyProjects\Python/NLP_Demo1\offen_words", 'r', encoding='utf-8')
        title = []
        word_list = []
        while True:
            line = file.readline()
            if not line:
                break
            t, w = line.split ("::")
            title.append (t)
            word_list.append (w)
        file.close ()

        path_save = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\wiki_model\wiki_corpus.model"
        # 加载训练好的模型wiki_corpus.model
        model = word2vec.Word2Vec.load (path_save)
        similary_6value = [0 for i in range (len (title))]
        for i in range (len (title)):
            try:
                similary_6value[i] = model.similarity (character, title[i])
            except Exception as e:
                character = character[0:2]
                try:
                    similary_6value[i] = model.similarity(character,title[i])
                except Exception as e:
                    character = character[0]

        similary_6value = np.array (similary_6value)
        max_6value_id = similary_6value.argsort()[::-1][0:before_key]
        word_6list = ""
        for i in range (before_key):
            word_6list += "".join (word_list[max_6value_id[i]])

        word_6list = word_6list.replace ("\n", "")
        word_6list = set (word_6list)
        return word_6list


if __name__ == '__main__':
    # path1..,path1_save_model ,生成各类诗的模型
    path1 = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir/biansai.txt'
    path1_save_model = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir/Keyword_biansai.model'
    path2 = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir/jingwu.txt'
    path2_save_model = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir/Keyword_jingwu.model'
    path3 = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir/shanshui.txt'
    path3_save_model = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir/Keyword_shanshui.model'
    path4 = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir/sixiang.txt'
    path4_save_model = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir/Keyword_sixiang.model'
    path5 = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir/yongshi.txt'
    path5_save_model = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir/Keyword_yongshi.model'
    # 训练每类的模型
    # Word2vec_similar.practice_model(path1, path1_save_model)
    # Word2vec_similar.practice_model (path2, path2_save_model)
    # Word2vec_similar.practice_model (path3, path3_save_model)
    # Word2vec_similar.practice_model (path4, path4_save_model)
    # Word2vec_similar.practice_model (path5, path5_save_model)

    # 依次对5类诗进行预处理
    # path1 = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\generate_poem\Poetry_class/yongshi.txt"
    # path2 = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir/yongshi.txt"
    # Word2vec_similar.file_deal(path1,path2)   # 语料 预处理

    # 某一类中，返 6个相似词， 关键词一定要在全唐诗中
    key_word = '边塞'
    # classtag = '边塞征战'
    # # 以固定类别加载 相似词 6 个
    # similar_keyword= Word2vec_similar.similar_6words(key_word,classtag)
    # print(similar_keyword)
    l = Word2vec_similar.class_tags(key_word)
    print (l)

    # 全唐诗相似词模型
    # Quanshi_path_save_model="E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir\Quanshi_Judge_6class.model"
    # 维基百科 2 词 相似度计算
    # wiki_path_model="E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\wiki_model\wiki_corpus.model"
    # 根据输入词，来判别哪一类
    # picture_path="E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir/yongshi.txt"
    # texts = open(picture_path,'r',encoding='utf-8').read()
    # texts = texts.replace('\n','')
    # Word2vec_similar.Picture_file(texts)
