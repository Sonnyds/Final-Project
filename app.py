from flask import Flask, render_template,  request
from flask_bootstrap import Bootstrap
from collections import Counter
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
import re
import numpy as np
import matplotlib.pyplot as plt
app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/index')
def home():
    return render_template("index.html")

@app.route('/movie')
def movie():
    data = pd.read_csv('movies.csv')
    movies_list = data.values.tolist()
    return render_template("movie.html", movies=movies_list)

@app.route('/score')
def score():
    data = pd.read_csv('movies.csv')
    score = data['scores'].tolist()
    score_counts = Counter(score)
    score_list = sorted(set(score))
    num_list = [score_counts[s] for s in score_list]
    return render_template("score.html", score=score_list, num=num_list)

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Get input variables from request
        year = request.form.get('year')
        year = float(year)
        runtime = request.form.get('runtime')
        runtime = float(runtime)
        genre = request.form.get('genre')
        directors = request.form.get('directors')
        stars = request.form.get('stars')

        data = pd.read_csv('movies.csv')
        data['runtime'] = data['runtime'].apply(lambda x: float(x.replace("min", "").strip()))

        # 电影上映年限数值化
        data["year"] = data["year"].apply(lambda x: float(re.findall(r'\d+', x)[0]))
        data["year"] = 2023. - data["year"]

        # 提取特征和目标变量
        X = data[['year', 'runtime', 'genre', 'directors', 'stars']]
        y = data['scores'].values
        bins = [6, 7.5, 9]

        # 使用digitize函数将分数划分为类别
        classes = np.digitize(y, bins)
        classes = classes.ravel()

        # 对分类变量进行 One-hot 编码
        enc = OneHotEncoder()
        X_cat = X[['genre', 'directors', 'stars']]
        X_cat_enc = pd.DataFrame(enc.fit_transform(X_cat).toarray(), columns=enc.get_feature_names_out())

        # 将编码后的分类特征和数值特征组合
        X_enc = pd.concat([X[['year', 'runtime']], X_cat_enc], axis=1)
        print(X_enc.head())

        # 拆分数据集为训练集和测试集

        # 训练决策树模型
        clf = DecisionTreeClassifier(max_depth=5,random_state=42)
        clf.fit(X_enc, classes)
        new_data = {'year': year, 'runtime': runtime, 'genre': genre, 'directors': directors, 'stars': stars}
        new_data_df = pd.DataFrame([new_data])
        # 对新数据进行特征编码
        new_data_cat = new_data_df[['genre', 'directors', 'stars']]
        new_data_cat_enc = pd.DataFrame(enc.transform(new_data_cat).toarray(), columns=enc.get_feature_names_out())
        new_data_enc = pd.concat([new_data_df[['year', 'runtime']], new_data_cat_enc], axis=1)

        # 使用训练好的模型进行预测
        predicted_class = clf.predict(new_data_enc)[0]
        if predicted_class == 1:
            popularity = 'Low Popularity'
        elif predicted_class == 2:
            popularity = 'Mid Popularity'
        else:
            popularity = 'High Popularity'


        # Return output variable
        return render_template('predict.html', popularity=popularity)

    return render_template('predict.html')

@app.route('/team')
def team():
    return render_template("team.html")








if __name__ == '__main__':

    app.run(debug=True)
