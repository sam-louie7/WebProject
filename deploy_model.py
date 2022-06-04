
import pickle
import pandas as pd
import numpy as np
from flask import Flask, redirect, render_template, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float
from traitlets import default
from generate_csv.single_session_data_extraction import generate_rawdata_df
from generate_features.emotional_rational_features import emo_ratio_single_session_hit_features, emo_ratio_single_session_tl_features


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tenen.db'
db = SQLAlchemy(app)
model = pickle.load(open('./models/active_model.pkl','rb'))

class Entry(db.Model):
    id_user = db.Column(db.String(50), primary_key=True)
    tag = db.Column(db.String(20), nullable=False)
    session_time = db.Column(db.Integer, default=0)
    page_views = db.Column(db.Integer, default=0)
    prod_views= db.Column(db.Float, default=0.0)

    def __repr__(self):
        return f'user-id: {self.id_user},  session time(sec): {self.session_time}, page veiwed: {self.page_views}, avg prod viewed: {self.prod_views}, pred:{self.tag}'


@app.route('/', methods=['GET'])
def query():
    query_res = Entry.query.all()
    return render_template("page.html", query_all=query_res)


@app.route('/', methods=['POST'])
def predict():
    f = request.files['textfile']
    json_path = './session/' + f.filename
    f.save(json_path)
    ## build df from row data 
    id, session_time, df = create_feature_df(json_path)
    label = "non-active"
    if session_time > 30:
        ## run data on trained model on feature vec
        pred = model.predict(df.values)[0]
        label = "active" if pred else "non-active"
    ## update db if active
    new_entry = Entry(id_user=id,
                      tag=label,
                      session_time=session_time,
                      page_views = df.avg_page_views[0],
                      prod_views = df.avg_viewd_product[0])
    try:
        db.session.add(new_entry)
        db.session.commit()
        classification = f' id: {id}    session(sec): {session_time}    page views: {df.avg_page_views[0]}    avg product views: {df.avg_viewd_product[0]}     pred: {label} '
        queryall = Entry.query.all()
        query_res = f'{queryall}'
        return render_template('page.html', prediction=classification, query_all=query_res)
    except:
        return 'there was an issue with your db'
    ## prepare classification
    

@app.route('/delete/', methods=['POST'])
def delete():
    id = request.form["delete"]
    task_to_delete = Entry.query.get_or_404(id)
    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        queryall = Entry.query.all()
        query_res = f'{queryall}'
        return render_template('page.html', query_all=query_res)
    except:
        return 'there was an issue with your db'


def create_feature_df(path):
    df_tl, df_hit = generate_rawdata_df(path)
    if 'totals.timeOnSite' in df_tl.columns.to_list():
        session_time = int(df_tl['totals.timeOnSite'][0])
    else:
        session_time = np.sum(df_hit['hits.deltaTimeMS']) / 1000.0
    id, df_f1 = emo_ratio_single_session_tl_features(df_tl=df_tl, session_time=session_time, numeric_cols=['totals.hits','totals.pageviews','totals.timeOnSite'])
    df_f2 = emo_ratio_single_session_hit_features(df_hit=df_hit, numeric_cols=['numOfProducts','hits.eCommerceAction.action_type','hits.time'])    
    df_features = pd.concat([df_f1,df_f2], axis=1)
    return id, session_time, df_features


if __name__ == '__main__':
    app.run(debug=True, port=3000)

