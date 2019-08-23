# Generate_Poems
Automatically write poems based on user keywords

Android端代码在目录：Android/LoginDemo/app/src/main/java/com/example/logindemo/
使用的是，AndroidStudio编程
Python端代码在Python目录下；
python端应用环境：
tensorflow
fasttext
flask
numpy
pandas

项目介绍链接：https://mp.csdn.net/postedit/92379563
项目的应用数据：
链接：https://pan.baidu.com/s/1xg7LPObdx2vbi28vqsjrwg 
提取码：1wg2 
data_dir目录，是5类诗歌生成此类诗的相似词的word2vec模型 + 数据
generate_poem目录，是用于生成5类诗歌的LSTM模型
train_jar目录，是诗歌前期fasttext分类模型 + 数据
wiki_model目录，是判别关键词所属类别的维基百科的word2vec模型
