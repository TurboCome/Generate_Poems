"""
    诗分类模型 ；Fasttext分类
    数据格式 ：'__label__'  + classtage(类型) + '\t'+ line（诗内容） + '\n'
"""
from config import*
# 训练数据统一格式
def file_deal(path,classtage):
    # 加载停用词
    stop_word = []
    stop_path = 'E:\Desk\MyProjects\Python/NLP_Demo1\CNstopwords.txt'
    with open(stop_path , 'r' ,encoding='utf-8') as stop_file:
        for line in stop_file:
            line = str(line.replace('\n','').replace('\r','').split())
            stop_word.append(line)
        stop_word = set(stop_word)

    path_name = path
    rules = u'[\u4e00-\u9fa5]+' #只是汉字的正则表达式，可以去除 ，。！（）等特殊字符
    pattern = re.compile(rules)
    sentences = []
    f_reader = open(path_name , 'r' , encoding = 'utf-8')
    while True:
        line = f_reader.readline()
        if not line:
            break
        line =line.replace('\n','').replace('\r','').split()
        line = str(line)
        line = ' '.join(jieba.cut(line))
        seg_list = pattern.findall(line)
        print(seg_list)
        word_list = []
        for word in seg_list:
            if word not in stop_word:
                word_list.append((word))
        if len(word_list) > 0 :
            sentences.append(word_list)
            line = " ".join(word_list)
            f_write=open('E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar\shi.txt','a+',encoding='utf-8')
            line2 = '__label__'  + classtage + '\t'+ line + '\n'
            f_write.write(line2)
            f_write.flush()
# 对数据进行训练产生模型   **.model文件
def fasttext_deal(tests):
    path ='E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar\shi.txt'
    # 生成模型
    model = fastText.train_supervised(
        input='E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar\shi.txt', epoch=25, lr=1.0,
        wordNgrams=2, verbose=2, minCount=1)
    # 保存模型
    path_save = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar/class_shi.model'
    model.save_model (path_save)
    # 用模型测试-返回labels 类别+概率
    # labels = model.predict(tests)
    # print(labels)


# 测试数据统一格式
def file_deal3(str_line):
    # 读取停用词
    stop_word = []
    with open ('E:\Desk\MyProjects\Python\Practice_word2vec\CNstopwords.txt', 'r', encoding='utf-8') as f_reader:
        for line in f_reader:
            line = str (line.replace ('\n', '').replace ('\r', '').split ())
            stop_word.append (line)
        stop_word = set (stop_word)
    # 文本预处理
    sentecnces = []
    rules = u'[\u4e00-\u9fa5]+'
    pattern = re.compile (rules)
    line = str_line
    line = line.replace ('\r', '').replace ('\n', '').split ()
    line = str(line)
    line = " ".join (jieba.cut (line))
    seg_list = pattern.findall(line)
    word_list = []
    for word in seg_list:
        if word not in stop_word:
            word_list.append (word)  # 去除 停用词
    if len (word_list) > 0:  # 去除 空行
        sentecnces.append (word_list)
        re_line = " ".join (word_list)  # 以空格来划分各个 词
    return re_line


if __name__ == '__main__':
    # 对各类训练数据进行统一格式处理
    # path1 = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar/biansai.txt'
    # classtage1 = '边塞征战'
    # file_deal (path1, classtage1)
    # path2 = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar/jingwu.txt'
    # classtage2 = '写景咏物'
    # file_deal (path2, classtage2)
    # path3 = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar/shanshui.txt'
    # classtage3 = '山水田园'
    # file_deal (path3, classtage3)
    # path4 = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar/sixiang.txt'
    # classtage4 = '思乡羁旅'
    # file_deal (path4, classtage4)
    # path5 = 'E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar/yongshi.txt'
    # classtage5 = '咏史怀古'
    # file_deal(path5,classtage5)

    # # 测试
    # 模拟测试 15首诗 分类
    # test_path = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/test_jar\zz_test.txt"
    # 5w+首诗， 分类测试
    test_path2 ="E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\data_dir\poetryTang.txt"
    test_path3 = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\generate_poem\Raw_poetrySong.txt"
    save_path = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar/train_jar\class_shi.model"
    model = fastText.load_model(save_path) # 加载模型
    # ******** 改变路径--》选择模拟测试分类， 真实测试分类
    f_reader = open(test_path3 , 'r',encoding='utf-8')
    # 写入 各类文件
    class_path1 = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\generate_poem\Poetry_class/biansai.txt"
    f_write1 = open(class_path1, 'a+',encoding='utf-8')
    class_path2 = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\generate_poem\Poetry_class/jingwu.txt"
    f_write2 = open(class_path2, 'a+',encoding='utf-8')
    class_path3 ="E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\generate_poem\Poetry_class/shanshui.txt"
    f_write3 = open(class_path3, 'a+',encoding='utf-8')
    class_path4 = "E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\generate_poem\Poetry_class/sixiang.txt"
    f_write4 = open(class_path4, 'a+',encoding='utf-8')
    class_path5 ="E:\Desk\MyProjects\Python/NLP_Demo1\File_jar\generate_poem\Poetry_class/yongshi.txt"
    f_write5 = open(class_path5, 'a+',encoding='utf-8')

    while True:
        line = f_reader.readline()
        if not line:
            break
        tests = file_deal3(line)
        # print(tests)
        # fasttext_deal(tests)
        label = model.predict(tests)  # 模型预测
        # label[0] 类别  label[1] 概率  label为元组
        value = label[0]
        print(label)
        value = str(value)
        # print(value)
        # print(type(label[0]))
        if operator.eq(value, "('__label__边塞征战',)")==True:
            f_write1.write(line)
            f_write1.flush()
        if operator.eq(value, "('__label__写景咏物',)")==True:
            f_write2.write(line)
            f_write2.flush()
        if operator.eq(value, "('__label__山水田园',)")==True:
            f_write3.write(line)
            f_write3.flush()
        if operator.eq(value, "('__label__思乡羁旅',)")==True:
            f_write4.write(line)
            f_write4.flush()
        if operator.eq(value, "('__label__咏史怀古',)")==True:
            f_write5.write(line)
            f_write5.flush()