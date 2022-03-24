# -*- coding: utf-8 -*-

from flask import Blueprint, render_template, flash, redirect, url_for, send_file, request, jsonify
from flask_login import login_required, current_user

from ..extensions import db, mongo

from flaskstarter.covid2k_meta import covid2k_metaModel
import csv

import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import json
from ..utils import TMP_FOLDER
import os
import shortuuid
import subprocess
import pandas as pd
import time
from datetime import datetime
tasks = Blueprint('tasks', __name__, url_prefix='/tasks')

# sub folders to manage different flask instances: not a good place to put, should be in API endpoint?
user_timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
user_tmp = [TMP_FOLDER + "/" + user_timestamp]
print(user_tmp)
os.makedirs(user_tmp[-1])


# New view
@tasks.route('/contribute')
def contribute():
    return render_template('tasks/contribute.html')


scClassify_input = []


@tasks.route('/uploader', methods = ['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(user_tmp[-1], uploaded_file.filename))
        scClassify_input.extend([uploaded_file.filename])
        flash('scClassify user file uploaded successfully.')

    return redirect(url_for('tasks.contribute'))


@tasks.route('/show_plot', methods=['GET', 'POST'])
def show_plot():
    graphJSON = plot_tse()
    #print(graphJSON)
    return render_template('tasks/show_plot.html', graphJSON=graphJSON)


def plot_tse():
    df = pd.read_csv(user_tmp[-1] + '/umap.csv', index_col=0)
    l = []
    for i in df.index:
        # print(df.loc[i].values)
        l.append(df.loc[i].values)
    sln = np.stack(l)
    projections = sln
    fig = px.scatter(
        projections, x=0, y=1)
    fig.update_layout(
        autosize=False, width=900, height=600
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@tasks.route('/run_scclassify', methods=['GET', 'POST'])
def run_scClassify():
    print("running scClassify...")
    scClassify_folder = TMP_FOLDER + "/scClassify"
    scClassify_code_path = scClassify_folder + "/scClassify_example_codes.r"
    scClassify_model_path = scClassify_folder + "/scClassify_trainObj_trainWillks_20201018.rds"
    scClassify_input_path = user_tmp[-1] + "/" + scClassify_input[-1]
    scClassify_output_path = user_tmp[-1] + "/scClassify_predicted_results.csv"
    ret = subprocess.call("Rscript %s %s %s %s" % (scClassify_code_path, scClassify_input_path, scClassify_model_path, scClassify_output_path), shell=True)
    if ret != 0:
        if ret < 0:
            print("Killed by signal")
        else:
            print("Command failed with return code %s" % ret)
    else:
        print("SUCCESS!!")
        flash("SUCCESS!!")
        return redirect(url_for('tasks.contribute'))


@tasks.route('/download_scClassify',methods=['POST'])
def download_scClassify():
    return send_file(user_tmp[-1] + "/scClassify_predicted_results.csv", as_attachment=True)


@tasks.route('/table_view')
def table_view():
    fsampleid = get_field("sample_id")
    fage = get_field("age")
    fdonor = get_field("donor")
    fprediction = get_field("scClassify_prediction")
    fstatus = get_field("Status_on_day_collection_summary")
    fdataset = get_field("dataset")
    return render_template('tasks/table_view.html',
                           fdonor=fdonor,
                           fage=fage,
                           fsampleid=fsampleid,
                           fprediction=fprediction,
                           fstatus=fstatus,
                           fdataset=fdataset)


#ids = [x["_id"] for x in list(db.single_cell_meta.find({}, {"_id": 1}))]

@tasks.route('/filter_by_age',methods=['POST'])
def filter_by_age():
    query = {'age': { '$in': [52] }}
    ids = [x["_id"] for x in list(mongo.single_cell_meta.find(query, {"_id": 1}))]
    print('return new ids')
    return api_db()


data = []


def write_file_bytype(folder, content, filetype):
    filename = folder + "/" + filetype + ".csv"
    if filetype == "id":
        towrite = [[r['id']] for r in content]
    elif filetype == "umap":
        towrite = [[r['id'], r['UMAP1'], r['UMAP2']] for r in content]
    elif filetype == "mtx":
        towrite = [[r['gene_name'], r['barcode'], r['expression']] for r in content]
    file = open(filename, 'w+', newline='\n')
    with file:
        print("Writing file %s" % filename)
        write = csv.writer(file)
        write.writerows(towrite)
    return filename


# Download big file
@tasks.route('/download_id',methods=['POST'])
def download_id():
    id = mongo.single_cell_meta.find({'_id': {'$in': collection_searched}}, {'id': 1, '_id': 0})
    filepath = write_file_bytype(user_tmp[-1], id, "id")
    return send_file(filepath, as_attachment=True)


@tasks.route('/download_umap', methods=['POST'])
def download_umap():
    fn = user_tmp[-1] + "/id.csv"
    _byid = pd.read_csv(fn).values.tolist()
    lookups = list(np.squeeze(_byid))
    umap = mongo.umap.find({'id': {'$in': lookups}})
    filepath = write_file_bytype(user_tmp[-1], umap, "umap")
    return send_file(filepath, as_attachment=True)


@tasks.route('/download_matrix',methods=['POST'])
def download_matrix():
    _byid = pd.read_csv(user_tmp[-1] + "/id.csv").values.tolist()
    lookups = list(np.squeeze(_byid))
    start_time2 = time.time()
    mtx = mongo.matrix.find({'id': {'$in': lookups}})
    print("--- %s seconds ---" % (time.time() - start_time2))
    filepath = write_file_bytype(user_tmp[-1], mtx, "mtx")
    return send_file(filepath, as_attachment=True)


# set in-memory storage for collection of ids for meta data table display
collection = []
collection_searched = []

# http://www.dotnetawesome.com/2015/12/implement-custom-server-side-filtering-jquery-datatables.html
@tasks.route('/api_db', methods=['GET', 'POST'])
@login_required
def api_db():
    data = []
    if request.method == 'POST':
        draw = request.form['draw']
        row = int(request.form['start'])
        rowperpage = int(request.form['length'])
        page_no = int(row/rowperpage + 1)
        search_value = request.form["search[value]"]
        print("draw: %s | row: %s | global search value: %s" % (draw, row, search_value))
        start = (page_no - 1)*rowperpage
        end = start + rowperpage
        map = {}
        for i in request.form:
            if ("[search][value]" in i) and (len(request.form[i]) != 0):
                column_value = request.form[i]
                if "0" in i:
                    search_column = "id"
                    map[search_column] = column_value
                elif "1" in i:
                    search_column = "sample_id"
                    map[search_column] = column_value
                elif "2" in i:
                    search_column = "age"
                    map[search_column] = column_value
                elif "3" in i:
                    search_column = "scClassify_prediction"
                    map[search_column] = column_value
                elif "4" in i:
                    search_column = "donor"
                    map[search_column] = column_value
                elif "5" in i:
                    search_column = "dataset"
                    map[search_column] = column_value
                else:
                    search_column = "Status_on_day_collection_summary"
                    map[search_column] = column_value
        print(map)
        if search_value == '':
            if len(collection) == 0:
                ids = [x["_id"] for x in mongo.single_cell_meta.find({}, {"_id": 1})]
                collection.extend(ids)
                tmp = mongo.single_cell_meta.find({'_id': {'$in': collection[start:end]}})
                total_records = len(collection)

            else:
                # refresh searchValue's stored ids when clicked "clear"
                collection_searched.clear()
                tmp = mongo.single_cell_meta.find({'_id': {'$in': collection[start:end]}})
                total_records = len(collection)

        else:
            if len(collection_searched) == 0:
                ids = [x["_id"] for x in mongo.single_cell_meta.find(json.loads(search_value), {"_id": 1})]
                collection_searched.extend(ids)
                tmp = mongo.single_cell_meta.find({'_id': {'$in': collection_searched[start:end]}})
                total_records = len(collection_searched)

            else:
                tmp = mongo.single_cell_meta.find({'_id': {'$in': collection_searched[start:end]}})
                total_records = len(collection_searched)

        if map:
            if len(collection_searched) == 0:
                print("using search id")
                construct = []
                for k in map:
                    if (k in ["age", "sample_id"]) and map[k].isdigit():
                        # if string only contains numbers, we need to convert it to integer to search
                        # age contains strings and integers, but the front-end will always process them to strings
                        q = {k: {"$in": [int(map[k])]}}
                        construct.append(q)
                    else:
                        q = {k: {"$in": [map[k]]}}
                        print(q)
                        construct.append(q)
                print(construct)
                ids = [x["_id"] for x in mongo.single_cell_meta.find({"$and": construct}, {"_id": 1})]
                collection_searched.extend(ids)
                tmp = mongo.single_cell_meta.find({'_id': {'$in': collection_searched[start:end]}})
                total_records = len(collection_searched)

            else:
                tmp = mongo.single_cell_meta.find({'_id': {'$in': collection_searched[start:end]}})
                total_records = len(collection_searched)
                # clear search result once new search input is clicked.
                collection_searched.clear()

        total_records_filter = total_records
        if total_records_filter == 0:
            print("return nothing")
            data.append({
                'id': "",
                'sample_id': "",
                'age': "",
                'prediction': "",
                'donor': "",
                'dataset': "",
                'status': ""
            })

        else:
            for r in tmp:
                data.append({
                        'id': r['id'],
                        'sample_id': r['sample_id'],
                        'age': r['age'],
                        'prediction': r['scClassify_prediction'],
                        'donor': r['donor'],
                        'dataset': r['dataset'],
                        'status': r['Status_on_day_collection_summary']
                    })

        response = {
                'draw': draw,
                'iTotalRecords': total_records,
                'iTotalDisplayRecords': total_records_filter,
                'aaData': data,
        }

        return jsonify(response)


# Old code for small sample dataset before server-side processing is introduced 2022.2.27
# Old logic
@tasks.route('/my_tasks', methods=['GET', 'POST'])
@login_required
def my_tasks(condition=''):
    print('debug...')
    print('change condition...')
    print(condition)
    graphJSON = {}

    if not condition:
        #_all_tasks = covid2k_metaModel.query.limit(10).all()
        _all_tasks = db.single_cell_meta.find({}).limit(10)
        #print(list(meta))
        #print(len(list(_all_tasks)))
        print('show default')

    else:
        if isinstance(condition, str):
            if condition == 'all':
                _all_tasks = covid2k_metaModel.query.all()
                writefile('/tmp/flaskstarter-instance/', _all_tasks)
                print('show all')
            else:
                print('show' + condition)
                # _all_tasks = db.session.execute('SELECT * FROM covid2k_meta WHERE age LIKE 52')
                _all_tasks = covid2k_metaModel.query.filter(covid2k_metaModel.age.contains(condition)).all()
                writefile('/tmp/flaskstarter-instance/', _all_tasks)
                graphJSON = plot()
        elif isinstance(condition, list):
            _all_tasks = covid2k_metaModel.query.filter(covid2k_metaModel.donor.in_(condition)).all()
            writefile('/tmp/flaskstarter-instance/', _all_tasks)
            graphJSON = plot()
            print('show donors')


    print("reload...")
    print(condition)



    return render_template('tasks/my_tasks.html',
                           all_tasks=_all_tasks,
                           graphJSON=graphJSON,
                           _active_tasks=True)

def writefile(path, towrite):
    print('writing data' + path)
    file = open(path + 'result.csv', 'w+', newline='\n')
    data = [[task.X] for task in towrite]
    with file:
        write = csv.writer(file)
        write.writerows(data)


@tasks.route('/showall',methods=['POST'])
def showall():
    return my_tasks('all')


@tasks.route('/age52',methods=['POST'])
def age_filter():
    return my_tasks('52')

@tasks.route('/download',methods=['POST'])
def download():
    plot()
    return send_file('/tmp/flaskstarter-instance/result.csv', as_attachment=True)


def get_field(field_name):
    key = mongo.single_cell_meta.distinct(field_name)
    #uniq_field = mongo.single_cell_meta.aggregate([{"$group": {"_id": '$%s' % field_name}}]);
    #key = [r['_id'] for r in uniq_field]
    print("%s has %d uniq fields" % (field_name, len(key)))
    return key


@tasks.route('/get_multiselect', methods=['POST'])
# uses list to store returned condition for my_tasks
def get_multiselect():
     selected_vals = request.form.getlist('multiselect')
     print(request.form)
     print(selected_vals)
     query = { 'donor': {'$in': selected_vals}}
     ids = [x["_id"] for x in list(mongo.single_cell_meta.find(query, {"_id": 1}))]
     #my_tasks(selected_vals)
     return table_view()

# to move outside of web app
def plot():
    path = '/tmp/flaskstarter-instance/'
    result = pd.read_csv(path + 'result.csv', header=None)
    df = pd.read_csv(path + 'cov192kaxis.csv', index_col=0)
    l = []
    for i in df.index:
        for j in result.values:
            if i == j:
                print(i)
                print(df.loc[i].values)
                l.append(df.loc[i].values)
    sln = np.stack(l)
    projections = sln
    fig = px.scatter(
        projections, x=0, y=1)
    fig.update_layout(
        autosize=False, width=900, height=600
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
