package com.example.logindemo;

import android.content.Intent;
import android.os.AsyncTask;
import android.os.Build;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

public class Register extends AppCompatActivity {
    EditText account_id;
    EditText password_id;
    EditText name_id ;
    EditText sex_id;
    EditText birth_id;


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
        setContentView(R.layout.activity_register);

        account_id = (EditText) findViewById(R.id.edit_account);
        password_id = (EditText)findViewById(R.id.edit_password);
        name_id = (EditText)findViewById(R.id.edit_name);
        sex_id = (EditText) findViewById(R.id.edit_sex);
        birth_id = (EditText) findViewById(R.id.edit_birth);

        Button must_register = (Button)findViewById(R.id.button_must_register);

        must_register.setOnClickListener(new View.OnClickListener() {



            @Override
            public void onClick(View v) {
                String account =account_id.getText().toString();
                String password = password_id.getText().toString();
                String name = name_id.getText().toString();
                String sex = sex_id.getText().toString();
                String birth = birth_id.getText().toString();

//                Toast.makeText(Register.this,account , Toast.LENGTH_SHORT).show();
                register(account,password,name,sex,birth);
            }
        });
    }


    protected void register(String account, String password,String name,String sex,String birth) {
        String registerUrlStr = Constant.URL_Register + "?userAccount=" + account + "&userPassword=" + password +"&userName="+
                name+ "&userSex="+sex + "&userBirth="+birth;
//        Toast.makeText(Register.this,registerUrlStr , Toast.LENGTH_SHORT).show();
        Log.d("JSON","指令："+registerUrlStr);
        Register.Login_AsyncTask login_asyncTask =new Register.Login_AsyncTask();
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
            Log.d("JSON",s);//打印服务器返回标签
            //flag=true;
            switch (s){
                //判断返回的状态码，并把对应的说明显示在UI
                case "注册成功":
                    Toast.makeText(Register.this,"注册成功",Toast.LENGTH_SHORT).show();
                    Intent intent = new Intent(Register.this,MainActivity.class);
                    startActivity(intent);
                    break;
                default:
                    Toast.makeText(Register.this,"注册失败,此账号已经存在",Toast.LENGTH_SHORT).show();
                    break;
                   }
            Log.d("JSON","验证后");
        }
    }

}
