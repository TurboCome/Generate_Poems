"""
    从古诗文网（https://www.gushiwen.org/）上爬取网页中的6类唐诗

"""
from config import *
def Get_url():
    # 改变标准输出的默认编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
    # 需要爬的网址--古诗文网https://www.gushiwen.org/shiwen/
    # 输入一类，翻页爬去，5类诗每类获取 600首，用作分类训练、测试数据
    for i in range(18,20):   # i 页面范围
        # 此网址总共有三种格式，经观察发现 page, A ,后跟随的便是页数，循环更改此值，便可实现翻页爬取
        # url='https://www.gushiwen.org/shiwen/default.aspx?page=9&type=4&id=1'
        # 'https://www.gushiwen.org/shiwen/default_1A589282347eb3A2.aspx'
        # 'https://so.gushiwen.org/search.aspx?type=title&page=3&value=%E7%BE%81%E6%97%85%E6%80%9D%E4%B9%A1'
        url1 = 'https://so.gushiwen.org/search.aspx?type=title&page='
        page = i
        page = '%d' % page
        url2 = '&value=%e5%8f%a4%e8%bf%b9'
        # '----------------------------------------------------------------------------------------------------------------'
        url=url1+page+url2  # 得到爬取具体网址
        head ={}
        # about:version  得到浏览器版本，用户代理，可实现伪装成浏览器
        head['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'
        req = request.Request(url, headers=head)
        response = request.urlopen(req)
        html = response.read()
        soup = BeautifulSoup(html, 'lxml')
        # print(soup)  # soup为所获取网页代码html

        poem = soup.find_all('textarea') # 得到<textarea>标签下的内容
        # print(poem[1])
        file=r'E:\Desk\MyProjects\Python\NLP_Demo1\File_jar/train_jar\yongshi.txt'
        # 获取textarea标签中的内容
        for i in soup.find_all(re.compile('textarea')):
            str=''.join(i.text) #列表--字符串
            str,_=str.rsplit('https')
            content,str2=str.rsplit('——')
            print(str2)
            # if str2=='近现代·伯昏子《题梦璧兄《羁旅吟》》':
            #     continue
            author,title=str2.rsplit('《')
            _,author=author.split('·')
            title,_=title.split('》')
            # print(content)
            # print(author)
            # print(title)
            with open(file,'a+') as f:
              f.write(title+'::'+author+'::'+content+'\n')# 以题目::作者::诗 的格式写入各个文件

    f.close()

if __name__ == '__main__':
    Get_url()

