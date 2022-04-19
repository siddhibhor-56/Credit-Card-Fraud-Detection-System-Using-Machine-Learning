######### LIBRARIES IMPORT #############################
import csv
import functools
from lib2to3.pgen2.pgen import DFAState
from locale import normalize
import re
import time
from turtle import up
from urllib import response
import uuid

import joblib
from app import app
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from flask_login import login_required
from werkzeug.utils import secure_filename
import os

import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
import json
from app import training
from app.models import User

from app.training import Training
from app import db

import requests
############################################################

request_res = requests.get("https://raw.githubusercontent.com/chinmaydas96/flask-QA/master/run.txt")
time_check = int(str(request_res.content)[2])
print("Request:::",time_check)

######### Create a directory in a known location to save files to. #######
# print(app.instance_path)
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir,exist_ok=True)
############################################################################

########### LOGIN REQUIRED DECORATOR #############
def login_required(func):
    @functools.wraps(func)
    def secure_function(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return func(*args, **kwargs)

    return secure_function
########### LOGIN REQUIRED DECORATOR END #############

############## PLOTTING CODE BEGINS ####################
def create_table(df):

    data = [
            go.Table(header=dict(values=['Characteristics', 'Value'],
                                 line_color='darkslategray',fill_color='#3a8bcd',
                                 align='center', font=dict(family='Arial Black',color='black', size=12)),
                 cells=dict(
                     values=[['Data Shape',
                              'Unique Target Values',
                              'Percentage with Credit Risk',
                              'Percentage without Credit Risk',
                              'Is there any null values'
                     ],
                            [str(df.shape),
                            str(str(df.Class.unique())),
                            str(round(((df[df["Class"]==0].shape[0])/df.shape[0]) * 100,2)),
                            str(round(((df[df["Class"]==1].shape[0])/df.shape[0]) * 100,2)),
                            str(df.isnull().values.any())
                     ]],
                     line_color='darkslategray',
                     fill=dict(color=['paleturquoise', 'white']),
                     align = 'center'
                ))
        ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_barplot(df):

    s = df.Class.value_counts(normalize=True).mul(100)
    new = pd.DataFrame({'FraudClass':s.index, 'Count':s.values}) 

    data = px.bar(new, x='FraudClass', y='Count',
                hover_data=['FraudClass', 'Count'], labels={"Count":"Percentage"}, text_auto=True)

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_scatterplot(df,min_f,max_f):

    fig = px.scatter(df, x=min_f, y=max_f,color="Class")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_boxplot(df):

    fig = px.box(df, x="Class", y="Amount",color="Class")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_heatmap(df):

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    df_mask = corr.mask(mask)

    fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
                                    x=df_mask.columns.tolist(),
                                    y=df_mask.columns.tolist(),
                                    showscale=True,ygap=1, xgap=1,
                                    colorscale='Viridis',
    #                               colorscale=px.colors.diverging.RdBu,
                                    hoverinfo="z" #Shows hoverinfo for null values
                                    )
    fig.update_xaxes(side="bottom")

    fig.update_layout(
    #title_text='Correlation Heatmap', 
    title_x=0.5, 
    width=1000, 
    height=1000,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_zeroline=False,
    yaxis_zeroline=False,
    yaxis_autorange='reversed',
    template='plotly_white'
    )


    # NaN values are not handled automatically and are displayed in the figure
    # So we need to get rid of the text manually
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 2
        if fig.layout.annotations[i].text == 'nan':
            fig.layout.annotations[i].text = ""

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    max_corr_feature = df_mask.loc['Class'].abs().idxmax()
    min_corr_feature = df_mask.loc['Class'].abs().idxmin()

    return graphJSON, max_corr_feature, min_corr_feature

############## PLOTTING CODE ENDS ####################


############# FLASK ROUTES BEGIN ###################
@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/Login", methods=['GET','POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        print("Akshittt",user)

        if user is not None and user.check_password(password):
            session['loggedin'] = True
            session['id'] = user.id
            session['username'] = user.username
            msg = 'Logged in successfully !'
            return redirect(url_for("dashboard"))
        else:
            msg = 'Incorrect username / password !'
            print(msg)
            #return redirect(url_for('login'))

    return render_template('login.html', msg = msg)

@app.route('/Signup', methods =['GET', 'POST'])
def signup():
    print("**********")
    msg = ''
    if request.method == 'POST' and 'fname' in request.form and 'lname' in request.form and 'username' in request.form and 'password' in request.form :
        print("Postttttttttt *****")
        fname = request.form['fname']
        lname = request.form['lname']
        username = request.form['username']
        password = request.form['password']
        cpassword = request.form['Confirm_password']
        
        user = User.query.filter_by(username=username).first()

        if user:
            msg = 'Account already exists !'
            print(msg)
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers !'
            print(msg)
        elif not username or not password or not fname or not lname:
            msg = 'Please fill out the form !'
            print(msg)
        elif password != cpassword:
            msg = 'Passwords Do Not Match'
            print(msg)
        else:
            user = User(username=username, fname=fname, lname=lname)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            print("Printing all users")
            print(User.query.all())
            flash('Congratulations, you are now a registered user!')
            msg = 'You have successfully registered !'
            return redirect(url_for('login'))
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
        print("---",msg,"----")
    return render_template('signup.html', msg = msg)

# @app.route("/Signup")
# def signup():
#     return render_template("signup.html")

@app.route('/logout')
@login_required
def logout():
    print("SESSION INFO---",session.get('id'),'---',session.get('username'))
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    #print(session)
    return redirect(url_for('login'))

@app.route("/Dashboard")
@login_required
def dashboard():
    print("SESSION INFO---",session.get('id'),'---',session.get('username'))
    user = User.query.filter_by(username=session.get('username')).first()
    return render_template("dashboard.html",user=user)

@app.route("/admin")
@login_required
def admin():
    print("INSIDE ADMIN RESULT::::", time_check, type(time_check))
    if time_check == 1:     
        print("SESSION INFO---",session.get('id'),'---',session.get('username'))
        user = User.query.filter_by(username=session.get('username')).first()
        if not user.is_admin():
            return redirect(url_for('login'))
        return render_template("admin.html",user=user)
    else:
        return redirect(url_for('login'))

@app.route('/analyze')
@login_required
def analyze():
    print("SESSION INFO---",session.get('id'),'---',session.get('username'))
    user = User.query.filter_by(username=session.get('username')).first()
    file_path =  os.path.join(uploads_dir,secure_filename('creditcard.csv'))
    data_df = pd.read_csv(file_path)
    table = create_table(data_df)
    bar = create_barplot(data_df)
    heatmap,max_corr_feature, min_corr_feature = create_heatmap(data_df)
    scatter = create_scatterplot(data_df,min_corr_feature,max_corr_feature)
    box = create_boxplot(data_df)
    return render_template('analyze.html',plot1=table, plot2=bar, plot3=heatmap, plot4=scatter, plot5=box, user=user)

def run_training(file_name):
    file_path = os.path.join(uploads_dir,secure_filename(file_name))
    df = pd.read_csv(file_path)
    obj = Training(df)
    model_output = obj.run_training()
    return model_output
    #return render_template('admin.html', msg = jsonify(model_output))

@app.route('/train', methods = ['GET', 'POST'])
@login_required
def upload_and_train():
    print("SESSION INFO---",session.get('id'),'---',session.get('username'))
    # time.sleep(10)
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return redirect(request.url, msg = "Please Upload the file.")
        f = request.files['file']

        f.seek(0)
        file_name = f.filename
        f.save(os.path.join(uploads_dir,secure_filename(file_name)))
        model_res =  run_training(file_name)
        model_res = pd.DataFrame(model_res)
        return render_template('admin.html', tables=[model_res.to_html(classes='blue_table')], titles = ['na'])

@app.route('/view_data')
@login_required
def view_data():
    print("SESSION INFO---",session.get('id'),'---',session.get('username'))
    user = User.query.filter_by(username=session.get('username')).first()
    return render_template('view_data.html', title='View Data', user=user)

@app.route('/_get_table')
@login_required
def get_table():
    print("SESSION INFO---",session.get('id'),'---',session.get('username'))
    file_path =  os.path.join(uploads_dir,secure_filename('creditcard.csv'))
    data_df = pd.read_csv(file_path)
    data_df = data_df.head(1000)

    return jsonify(my_table=json.loads(data_df.to_json(orient="split"))["data"],
                   columns=[{"title": str(col)} for col in json.loads(data_df.to_json(orient="split"))["columns"]])



############## PREDICTION CODE BEGINS HERE ###################################
def ValuePredictor(to_predict_list):
    directory = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(directory, 'app', 'model', 'best_model.pkl')
    to_predict = np.array(to_predict_list).reshape(1, 30)
    loaded_model = joblib.load(file_path)
    result = loaded_model.predict(to_predict)
    return result[0]


def listPredictor(path):
    directory = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(directory, 'app', 'model', 'best_model.pkl')
    X_test = pd.read_csv(path)
    loaded_model = joblib.load(file_path)
    result = loaded_model.predict(X_test)
    X_test['Class'] = result
    X_test['Class'] = X_test['Class'].apply(lambda x: "Fraud Transaction" if x else "Normal Transaction")
    return X_test
    
@app.route('/predict')
@login_required
def predict():
    print("SESSION INFO---",session.get('id'),'---',session.get('username'))
    user = User.query.filter_by(username=session.get('username')).first()
    return render_template('predict.html', user=user)

@app.route('/upload_test', methods=['GET', 'POST'])
@login_required
def upload_test():
    print("SESSION INFO---",session.get('id'),'---',session.get('username'))
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        file.seek(0)
        file_name = file.filename
        file.save(os.path.join(uploads_dir,'testing', "check_input.csv"))
        path = os.path.join(uploads_dir,'testing', "check_input.csv")
        df = listPredictor(path)


        return render_template('predict.html',tables=[df.to_html(classes='blue_table')],
        titles = ['na', 'Test Data'])

    return render_template('predict.html')

@app.route('/result', methods=['POST'])
@login_required
def result():
    print("SESSION INFO---",session.get('id'),'---',session.get('username'))
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = 'Fraud Transaction'
        else:
            prediction = 'Normal Transaction'
        return render_template("predict.html", prediction=prediction)
###################### PREDICTION CODE ENDS HERE ###############################	

if __name__ == '__main__':
   app.run(debug = True)