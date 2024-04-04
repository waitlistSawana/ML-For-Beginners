import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__) # 生成实例

model = pickle.load(open("../ufo-model.pkl", "rb")) # 导入模型


@app.route("/") 
def home():
    return render_template("index.html") # 渲染网页


@app.route("/predict", methods=["POST"]) # 
def predict():

    int_features = [int(x) for x in request.form.values()] # 接收数据参数
    final_features = [np.array(int_features)] # 
    prediction = model.predict(final_features) # 应用数据生成了预测结果

    output = prediction[0] # 第一个是结果

    countries = ["Australia", "Canada", "Germany", "UK", "US"] # 国家列表

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output]) # 生成网页 预测结果为
    )


if __name__ == "__main__":
    app.run(debug=True) # 测试用参数 生成错误信息 生产时删除