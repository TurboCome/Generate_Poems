package com.example.logindemo;

public class User {

    private int id;//序号
    private String account;//手机号码
    private String name;//昵称
    private String password;//密码
    private String sex;//性别
    private String birth;//生日

    public User(){

    }

    public String Setuser(){

        return "id:"+id+",账户："+account+",昵称："+name+",密码："+password+",性别："+sex+",生日："+birth;
    }
    public int getId() {
        return id;
    }
    public void setId(int id) {
        this.id = id;
    }


    public String getAccount() {
        return account;
    }
    public void setAccount(String account) {
        this.account = account;
    }


    public String getName() {
        return name;
    }
    public void setName(String name) {
        this.name = name;
    }


    public String getPassword() {
        return password;
    }
    public void setPassword(String password) {
        this.password = password;
    }


    public String getSex() {
        return sex;
    }
    public void setSex(String sex) {
        this.sex = sex;
    }


    public String getBirth() {
        return birth;
    }
    public void setBirth(String birth) {
        this.birth = birth;
    }



}
