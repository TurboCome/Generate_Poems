package com.example.logindemo;

import android.content.Intent;
import android.os.Build;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import android.os.AsyncTask;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.alibaba.fastjson.JSON;

import org.w3c.dom.Text;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;



public class MainActivity extends AppCompatActivity{


    EditText account;
    EditText password;

    Button login;
    Button register;
    Button user_list;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
//去掉顶部标题
        getSupportActionBar().hide();
//去掉最上面时间、电量等
//        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        if(Build.VERSION.SDK_INT>= Build.VERSION_CODES.KITKAT) {
            //透明状态栏
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
            // 透明导航栏
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_NAVIGATION);
        }

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        account=(EditText)findViewById(R.id.account);//输入账户
        password=(EditText)findViewById(R.id.password);//输入密码

        login=(Button)findViewById(R.id.login);       // 登陆
        register=(Button)findViewById(R.id.register);  //注册
        login.getBackground().setAlpha(90);  //0~255透明度值
        register.getBackground().setAlpha(90);
//        user_list=(Button)findViewById(R.id.user_list);  // 用户信息

        login.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!(account.getText().toString().isEmpty())
                        && !(password.getText().toString().isEmpty())) {
                    login(account.getText().toString(), password.getText().toString());

//                    Toast.makeText(MainActivity.this,"登陆成功" ,Toast.LENGTH_SHORT).show();

                } else {
                    Toast.makeText(MainActivity.this, "账号、密码都不能为空！", Toast.LENGTH_SHORT).show();
                }
                   }
        });

        register.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                    // 进入注册界面
                    Intent intent = new Intent(MainActivity.this,Register.class);
                    startActivity(intent);
            }
        });

        }



    private void login(String account, String password) {
        String registerUrlStr = Constant.URL_Login + "/?userAccount=" + account + "&userPassword=" + password;
//        Toast.makeText(MainActivity.this,registerUrlStr,Toast.LENGTH_SHORT).show();
        Log.d("JSON","指令："+registerUrlStr);
        Login_AsyncTask login_asyncTask =new Login_AsyncTask();
        login_asyncTask.execute(registerUrlStr);
    }

    class Login_AsyncTask extends AsyncTask<String, Integer, String> {

        public Login_AsyncTask() {
            Log.d("JSON","验证前");
        }

        @Override
        public void onPreExecute() {
            Log.w("JSON", "开始验证.........");
        }

        /**
         * @param params 这里的params是一个数组，即AsyncTask在激活运行是调用execute()方法传入的参数
         */
        @Override
        public String doInBackground(String... params) {
            HttpURLConnection connection = null;
            StringBuilder response = new StringBuilder();
            try {
                URL url = new URL(params[0]); // 声明一个URL
                connection = (HttpURLConnection) url.openConnection(); // 打开该URL连接
                connection.setRequestMethod("GET"); // 设置请求方法，“POST或GET”，我们这里用GET，在说到POST的时候再用POST
                connection.setConnectTimeout(80000); // 设置连接建立的超时时间
                connection.setReadTimeout(80000); // 设置网络报文收发超时时间
                BufferedReader reader =new BufferedReader(new InputStreamReader(connection.getInputStream()));
                String line;
                while ((line = reader.readLine()) != null) {
                    response.append(line);
                }
            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
            Log.d("JSON",response.toString()+"dsdfasfdfedfrwfrwf");
            return response.toString(); // 这里返回的结果就作为onPostExecute方法的入参
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            // 如果在doInBackground方法，那么就会立刻执行本方法
            // 本方法在UI线程中执行，可以更新UI元素，典型的就是更新进度条进度，一般是在下载时候使用
        }

        /**
         * 运行在UI线程中，所以可以直接操作UI元素
         * @param s
         */
        @Override
        protected void onPostExecute(String s) {
            Log.d("JSON",s);  //打印服务器返回标签
//            Toast.makeText(MainActivity.this, s , Toast.LENGTH_SHORT).show();
            //flag=true;
            switch (s){
                  //判断返回的状态码，并把对应的说明显示在UI
                case "登陆成功":
                    Toast.makeText(MainActivity.this,"登录成功",Toast.LENGTH_SHORT).show();
                    Intent intent = new Intent(MainActivity.this,WritePoem.class);
                    startActivity(intent);
                    break;
                default:
                    Toast.makeText(MainActivity.this,"登陆失败，账号或密码错误",Toast.LENGTH_SHORT).show();
                    break;
            }
            Log.d("JSON","验证后");
        }
    }
}
