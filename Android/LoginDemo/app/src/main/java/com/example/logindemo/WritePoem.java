package com.example.logindemo;

import android.os.AsyncTask;
import android.os.Build;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

public class WritePoem extends AppCompatActivity {

    EditText table;
    String lable ; // 用户输入关键词
    Button five_jue;
    Button seven_jue;
    Button five_lv;
    Button seven_lv;
    // 八句 诗
    TextView textpoem1;
    TextView textpoem2;
    TextView textpoem3;
    TextView textpoem4;
    TextView textpoem5;
    TextView textpoem6;
    TextView textpoem7;
    TextView textpoem8;


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
        setContentView(R.layout.activity_write_poem);

//       写诗文本栏
        textpoem1 = (TextView)findViewById(R.id.poem_content1);
        textpoem2 = (TextView)findViewById(R.id.poem_content2);
        textpoem3 = (TextView)findViewById(R.id.poem_content3);
        textpoem4 = (TextView)findViewById(R.id.poem_content4);
        textpoem5 = (TextView)findViewById(R.id.poem_content5);
        textpoem6 = (TextView)findViewById(R.id.poem_content6);
        textpoem7 = (TextView)findViewById(R.id.poem_content7);
        textpoem8 = (TextView)findViewById(R.id.poem_content8);

        table = (EditText)findViewById(R.id.edit_label);

        five_jue = (Button)findViewById(R.id.button_five_jue);
        five_jue.getBackground().setAlpha(90);
        five_jue.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                lable = table.getText().toString();
                String userNum = "5";
                String userJueLv = "4";
                PoemKinds(userNum , userJueLv,54,lable);
//                Toast.makeText(WritePoem.this,lable,Toast.LENGTH_SHORT).show();
            }
        });

        five_lv = (Button)findViewById(R.id.button_five_lv);
        five_lv.getBackground().setAlpha(90);
        five_lv.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                lable = table.getText().toString();
                String userNum = "5";
                String userJueLv = "8";
                PoemKinds(userNum,userJueLv,58,lable);
            }
        });

        seven_jue = (Button) findViewById(R.id.button_seven_jue);
        seven_jue.getBackground().setAlpha(90);
        seven_jue.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                lable = table.getText().toString();
                String userNum = "7";
                String userJueLv = "4";
                PoemKinds(userNum,userJueLv,74,lable);
            }
        });

        seven_lv = (Button) findViewById(R.id.button_seven_lv);
        seven_lv.getBackground().setAlpha(90);
        seven_lv.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                lable = table.getText().toString();
                String userNum = "7";
                String userJueLv = "8";
                PoemKinds(userNum,userJueLv,78,lable);
            }
        });

    }

    protected void PoemKinds(String Num,String JueLv,int url,String keyword) {
        String registerUrlStr;
        switch (url){
            case 54:{
                registerUrlStr = Constant.URL_Five_Jue + "?userNum=" + Num + "&userJueLv=" + JueLv+"&userKeyword="+keyword;
                break;
            }
            case 58:{
                registerUrlStr = Constant.URL_Five_Lv + "?userNum=" + Num + "&userJueLv=" + JueLv+"&userKeyword="+keyword;
                break;
            }
            case 74:{
                registerUrlStr = Constant.URL_Seven_Jue + "?userNum=" + Num + "&userJueLv=" + JueLv+"&userKeyword="+keyword;
                break;
            }
            case 78:{
                registerUrlStr = Constant.URL_Seven_Lv + "?userNum=" + Num + "&userJueLv=" + JueLv+"&userKeyword="+keyword;
                break;
            }
            default: {  // 默认为 --> 五言绝句
                registerUrlStr = Constant.URL_Five_Jue + "?userNum=" + Num + "&userJueLv=" + JueLv+"&userKeyword="+keyword;
                break;
            }
        }

//        Toast.makeText(WritePoem.this, registerUrlStr , Toast.LENGTH_SHORT).show();
        Log.d("JSON","指令："+registerUrlStr);
        WritePoem.Login_AsyncTask login_asyncTask =new WritePoem.Login_AsyncTask();
        login_asyncTask.execute(registerUrlStr);
    }


    class Login_AsyncTask extends AsyncTask<String, Integer, String> {

        public Login_AsyncTask() {
            Log.d("JSON", "验证前");

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
                BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));
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
         *
         * @param s
         */
        @Override
        protected void onPostExecute(String s) {
            Log.d("JSON", s);//打印服务器返回标签
            // s 为返回来的 诗
            textpoem1.setText("");
            textpoem2.setText("");
            textpoem3.setText("");
            textpoem4.setText("");
            textpoem5.setText("");
            textpoem6.setText("");
            textpoem7.setText("");
            textpoem8.setText("");

            String line1 [] = new String [2];
            String line2 [] = new String [2];
            String line3 [] = new String [2];
            String line4 [] = new String [2];

            String poems [] =s.split("。");
            int l = poems.length;  // 确定是 几句 诗

            line1= poems[0].split(",");
            line2= poems[1].split(",");
            Log.d("POEM0" , poems[0]);
            Log.d("POEM1" , poems[1]);
            textpoem1.setText(line1[0]);
            textpoem2.setText(line1[1]);
//            Log.d("Po",line1[0]);
            textpoem3.setText(line2[0]);
            textpoem4.setText(line2[1]);
            // 若是律诗，继续写 后四句
            if(l > 2){
                line3= poems[2].split(",");
                line4= poems[3].split(",");
                Log.d("POEM2" , poems[2]);
                Log.d("POEM3" , poems[3]);
                textpoem5.setText(line3[0]);
                textpoem6.setText(line3[1]);
//                Log.d("Po",line3[0]);
                textpoem7.setText(line4[0]);
                textpoem8.setText(line4[1]);
            }

        }
    }
}