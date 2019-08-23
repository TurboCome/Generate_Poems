import pymysql
# 封装数据库连接类
class SunckSql():
    def __init__(self,host , user, passwd , dbName):
        self.host = host  # 主机IP地址
        self.user = user  # 登陆数据库的用户名
        self.passwd = passwd # 密码
        self.dbName = dbName #所用数据库名称

    def connet(self):  # 连接MySQL数据库
        self.db = pymysql.connect(self.host,self.user,self.passwd,self.dbName)
        self.cursor = self.db.cursor()

    def close(self):  # 关闭连接
        self.cursor.close()
        self.db.close()

    def get_one(self , sql):  # 查询表中满足条件的下一条信息
        res = None
        try:
            self.connet()
            self.cursor.execute(sql) # 执行sql语句
            res =self.cursor.fetchone()
            self.close()
        except:
            print("查询失败！")
        return res

    def get_all(self ,sql):# 查询表中满足条件的所有信息
        res = None
        try:
            self.connet()
            #使用连接对象获得一个cursor对象,接下来,我们会使用cursor提供的方法
            self.cursor.execute(sql)
            res= self.cursor.fetchall()
            self.close()
        except:
            print("查询失败！")
        return res

    def insert(self , sql):
        return self.__edit(sql)

    def update(self , sql):
        return self.__edit(sql)

    def delete(self , sql):
        return self.__edit(sql)

    def __edit(self, sql):
        res = None
        try:
            self.connet()
            res = self.cursor.execute(sql)
            #执行完插入或删除或修改操作后,需要调用一下conn.commit()方法进行提交.这样,数据才会真正保 存在数据库中
            self.db.commit()
            self.close()
        except:
            print("事务提交失败！")
            self.db.rollback()
        return res
