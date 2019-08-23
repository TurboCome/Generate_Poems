"""
    对获取的唐诗文件进行预处理--去除 【】（），并统一 训练格式
"""

file= r'E:/Desk/MyProjects/Python/NLP_Demo1/File_jar/train_jar//jingwu.txt'
def dealone():
    file1= open('E:/Desk/MyProjects/Python/NLP_Demo1/File_jar/train_jar/jingwu.txt','r',errors='ignore')
    # class1='biansai'
    while True:
        mystr = file1.readline()
        if not mystr:
            break
        print(mystr)
        while True:
            if mystr.find ('(') != -1:
                first_in, mystr = mystr.split ("(", 1)
                _, end_in = mystr.split (')', 1)
                mystr = first_in + end_in
            else:
                break

        while True:
            if mystr.find ('[') != -1:
                first_in, mystr = mystr.split ('[', 1)
                _, end_in = mystr.split (']', 1)
                mystr = first_in + end_in
            else:
                break

        with open(file,'a+') as f:
            f.write(mystr)
    file1.close()


def dealtwo():
    file2 = open('E:/Desk/MyProjects/Python/NLP_Demo1/File_jar/train_jar/jingwu.txt','r',errors='ignore')
    class2='jingwu'
    while True:
        mystr2 = file2.readline()
        if not mystr2:
            break
        with open(file,'a+') as f:
            f.write(class2+'    '+mystr2)
    file2.close()

def dealthree():
    file3 = open('E:/Desk/MyProjects/Python/NLP_Demo1/File_jar/train_jar/shanshui.txt','r',errors='ignore')
    class3 = 'shanshui'
    while True:
        mystr = file3.readline()
        if not mystr:
            break
        with open(file,'a+') as f:
            f.write(class3 +'   '+mystr)
    file3.close()

def dealfour():
    file4 = open('E:/Desk/MyProjects/Python/NLP_Demo1/File_jar/train_jar/sixiang.txt','r',errors='ignore')
    class4='sixiang'
    while True:
        mystr=file4.readline()
        if not mystr:
            break
        with open(file,'a+') as f:
            f.write(class4+'    '+mystr)
    file4.close()

def dealfive():
    file5= open('E:/Desk/MyProjects/Python/NLP_Demo1/File_jar/train_jar/yongshi.txt','r')
    class5='yongshi'
    while True:
        mystr=file5.readline()
        if not mystr:
            break
        with open(file,'a+') as f:
            f.write(class5 + '  '+mystr)
    file5.close()


def little_word_deal():

    pass

if __name__ == '__main__':
    # dealone()
    # dealtwo()
    # dealthree()
    # dealfour()
    # dealfive()
    # print()

    little_word_deal()
