from flask import Flask,request
from SQL_operator import SunckSql

import config
import main
app = Flask(__name__)
global Num,JueLv,Keyword
global s ,ipv4

ipv4 ='192.168.43.244'
s = SunckSql(ipv4 ,'root','123456','practice')  # practice 数据库名称
# 注册：
@app.route("/LoginDemo/RegisterServlet/",methods=["GET"])
def register_servlet():
    print(request.form)
    print(request.args)
    # print(request.args['userAccount'] + "\n" + request.args['userPassword']+'\n'+
    #       request.args['userName']+'\n'+request.args['userSex']+'\n'+request.args['userBirth'])
    account = request.args['userAccount']
    password = request.args['userPassword']
    uname = request.args['userName']
    print(uname)
    sex = request.args['userSex']
    birth = request.args['userBirth']

    sql = "Insert into userinforma VALUES(" + account+","+password+",'"+uname+"','"+ sex+"','"+birth+"')"
    print(sql)
    result = s.insert(sql)
    if result != None:
        return "注册成功"
    else:
        return "注册失败"

# 登陆
@app.route("/LoginDemo/LoginServlet/",methods=["GET" ])
def login_servlet():
    print(request.args)
    print(request.args['userAccount'] + "  "+ request.args['userPassword'])
    account = request.args['userAccount']
    password = request.args['userPassword']
    # s = SunckSql("127.0.0.1", 'root','123456','practice')
    sql = "Select userAccount from userinforma where userAccount =" + account+" and userPassword= "+ password
    print(sql)
    result = s.get_one(sql)
    if result != None:

        return "登陆成功"
    else:
        return "登陆失败"


# 传送 诗类型 几言 - 绝句/律诗 + 关键词
@app.route("/LoginDemo/",methods=["GET"])
def poem_kinds():
    # print(request.args)
    # print(request.args['userNum'] + "   "+ request.args['userJueLv']+ " " + request.args['userKeyword'])
    # 根据用户输入得到 五/七言+ 绝句/律诗 + 关键词
    Num = request.args['userNum']
    JueLv= request.args['userJueLv']
    Keyword = request.args['userKeyword']

    # return num,juelv,keyword
    poems = main.Main.Finalsummary(Num, JueLv, Keyword)

    # print('\n'+poems)
    return poems

if __name__ == "__main__":
    app.run(host= ipv4, port=8080)
