# -*- coding: utf-8 -*-

from flask import Blueprint, render_template, flash, redirect, url_for, send_file, request, jsonify
from flask_login import login_required, current_user
from ..extensions import db, mongo

from flaskstarter.tasks.forms import UmapForm

import csv
import gzip
import zipfile
import re

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
from os.path import exists
naso = Blueprint('naso', __name__, url_prefix='/naso')

# sub folders to manage different flask instances: not a good place to put, should be in API endpoint?
user_timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
#user_timestamp = "test"
user_tmp = [TMP_FOLDER + "/" + user_timestamp]
print(user_tmp)
os.makedirs(user_tmp[-1])

scClassify_input = []




@naso.route('/show_plot', methods=['GET', 'POST'])
def show_plot():     
    cell_color = request.form.get('name_opt_col')
    if(cell_color is None):
        cell_color = "scClassify_prediction"

    graphJSON, df_plot = plot_umap(cell_color)
    colors = df_plot.columns.values
    return render_template('tasks/show_plot.html', graphJSON=graphJSON, colors=colors)


## Create by junyi
def plot_umap(cell_color='scClassify_prediction'):
    df = pd.read_csv(user_tmp[-1] + '/umap.csv', index_col=0)
    df.columns = ["umap_0", "umap_1"]
    df_meta = pd.read_csv(user_tmp[-1] + '/meta.tsv', index_col=1,sep="\t")
    df_plot = df.merge(df_meta, left_index=True, right_index=True)

    fig = px.scatter(
        df_plot, x="umap_0", y="umap_1",color=cell_color)
    fig.update_layout(
        autosize=False, width=900, height=600
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON, df_plot


@naso.route('/naso_tableview')
def naso_tableview():
    fsampleid = get_field("meta_sample_id2")
    fage = get_field("meta_age")
    print(fage)
    fdonor = get_field("meta_patient_id")
    fprediction = get_field("level2")
    fstatus = get_field("meta_severity")
    fdataset = get_field("meta_dataset")
    return render_template('tasks/naso_tableview.html',
                           fdonor=fdonor,
                           fage=fage,
                           fsampleid=fsampleid,
                           fprediction=fprediction,
                           fstatus=fstatus,
                           fdataset=fdataset)


#ids = [x["_id"] for x in list(db.single_cell_meta.find({}, {"_id": 1}))]

data = []


# Write big file
def write_file_byid(path, towrite):
    fn = path + '/ids.csv'
    print('writing ids to' + fn)
    file = open(fn, 'w+', newline='\n')
    #print(towrite[0])
    data = [[r['id']] for r in towrite]
    #print(data)
    with file:
        write = csv.writer(file)
        write.writerows(data)


# Write file
def write_file_meta(path, towrite):
    fn = path + '/meta.tsv'
    print('writing meta to' + fn)
    print(towrite[0])
    with open(fn, 'w') as file:
        file.write("\t".join([str(e) for e in towrite[0].keys()]))
        file.write('\n')

        for line in towrite:
            file.write("\t".join([str(e) for e in line.values()]))
            file.write('\n')


def write_umap(path, towrite):
    fn = path + '/umap.csv'
    print('writing umap to' + fn)
    file = open(fn, 'w+', newline='\n')
    #print(towrite[0])
    data = [[r['id'], r['UMAP1'], r['UMAP2']] for r in towrite]
    #print(data)
    with file:
        print("Writing file %s" % fn)
        write = csv.writer(file)
        write.writerows(data)
    return fn


# Write mtx is not okay, mem usage is too large and it will crash the server.
# def write_mtx(path, towrite):
#     fn = path + '/mtx.csv'
#     file = open(fn, 'w+', newline='\n')
#     print('writing mtx to' + fn)
#     #print(towrite[0])
#     data = [[r['gene_name'], r['barcode'], r['expression']] for r in towrite]
#     #print(data)
#     with file:
#         write = csv.writer(file)
#         write.writerows(data)
def write_10x_mtx(path,gene_dict,barcode_dict,counts, towrite):

    fn = path + '/matrix.mtx.gz'
    print('writing matrix.mtx.gz to' + fn)

    ##text=List of strings to be written to file
    with gzip.open(fn,'wb') as file:
        file.write("%%MatrixMarket matrix coordinate real general".encode())
        file.write('\n'.encode())
        file.write(" ".join([str(len(gene_dict)), str(len(barcode_dict)), str(counts)]).encode())
        file.write('\n'.encode())
        for line in towrite:
            file.write(" ".join([
                str(gene_dict[line['gene_name']]),
                str(barcode_dict[line['barcode']]),
                str(line['expression']),
            ]).encode())
            file.write('\n'.encode())


# Download big file
@naso.route('/download_meta', methods=['POST'])
def download_meta():
    meta = list(mongo.single_cell_meta.find({'_id': {'$in': collection_searched}}))
    print('writing ids to csv file only once, firstly load the data')
    write_file_byid(user_tmp[-1], meta)
    write_file_meta(user_tmp[-1], meta)
    #return send_file(user_tmp[-1] + '/ids.csv', as_attachment=True)
    return send_file(user_tmp[-1] + '/meta.tsv', as_attachment=True)


@naso.route('/download_umap', methods=['POST'])
def download_umap():
    f = user_tmp[-1] + "/ids.csv"
    if not os.path.isfile(f):
        id = mongo.single_cell_meta.find({'_id': {'$in': collection_searched}}, {'id': 1, '_id': 0})
        write_file_byid(user_tmp[-1], id)
        print(f)

    _byid = pd.read_csv(f).values.tolist()
    lookups = list(np.squeeze(_byid))
    umap = mongo.umap.find({'id': {'$in': lookups}})
    write_umap(user_tmp[-1], umap)
    return send_file(user_tmp[-1] + '/umap.csv', as_attachment=True)


@naso.route('/download_matrix', methods=['POST'])
def download_matrix():
    # Down load 10x matrix if not exist
    if(not (exists(user_tmp[-1] + '/matrix.mtx.gz'))):
        _byid = pd.read_csv(user_tmp[-1] + "/ids.csv").values.tolist()
        lookups = list(np.squeeze(_byid))
        start_time2 = time.time()
        mtx = mongo.matrix.find({'barcode': {'$in': lookups}})
        #query_counts = mtx.count()
        mtx = list(mtx)

        print("query finished --- %s seconds ---" % (time.time() - start_time2))

        ## Parse the barcode and gene based on name 
        def get_dict(path, sep="\t", header=None, save_path=None):
            # Transfrom the gene/barcode name to the corresponding number
            df_read = pd.read_csv(path, sep=sep, header=header)
            row_num = [i for i in range(1, len(df_read)+1)]
            row_name = list(df_read.iloc[:, 0].values)
            result_dict = dict(zip(row_name, row_num))
            if(not(save_path is None)):
                df_read.to_csv(save_path, sep="\t", header=False, index=False, compression='gzip')
            return result_dict

        # Save gene name
        if(not (exists(user_tmp[-1] + '/genes.tsv.gz'))):
            dict_gene = get_dict(TMP_FOLDER + "/genes.tsv", save_path=user_tmp[-1] + "/genes.tsv.gz")
        # Save barcodes
        if(not (exists(user_tmp[-1] + '/barcodes.tsv.gz'))):
            dict_barcode = get_dict(user_tmp[-1] + "/ids.csv", sep=",", save_path=user_tmp[-1] + "/barcodes.tsv.gz")
        write_10x_mtx(user_tmp[-1], dict_gene, dict_barcode, len(mtx), mtx)

    if(not (exists(user_tmp[-1] + '/matrix.zip'))):
        list_files = [
            user_tmp[-1] + '/matrix.mtx.gz',
            user_tmp[-1] + '/genes.tsv.gz',
            user_tmp[-1] + '/barcodes.tsv.gz',
            user_tmp[-1] + '/meta.tsv'
        ]
        with zipfile.ZipFile(user_tmp[-1] + '/matrix.zip', 'w') as zipMe:        
            for file in list_files:
                zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)

    return send_file(user_tmp[-1] + '/matrix.zip', as_attachment=True)


# set in-memory storage for collection of ids for meta data table display
collection = []
collection_searched = []

# http://www.dotnetawesome.com/2015/12/implement-custom-server-side-filtering-jquery-datatables.html
@naso.route('/api_db', methods=['GET', 'POST'])
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
        print(request.form)
        start = (page_no - 1)*rowperpage
        end = start + rowperpage
        map = {}
        for i in request.form:
            if ("[search][value]" in i) and (len(request.form[i]) != 0):
                column_value = request.form[i].split("|")
                if "0" in i:
                    search_column = "id"
                    map[search_column] = column_value
                elif "1" in i:
                    search_column = "meta_sample_id"
                    map[search_column] = column_value
                elif "2" in i:
                    search_column = "meta_age"
                    map[search_column] = column_value
                elif "3" in i:
                    search_column = "level2"
                    map[search_column] = column_value
                elif "4" in i:
                    search_column = "meta_patient_id"
                    map[search_column] = column_value
                elif "5" in i:
                    search_column = "meta_dataset"
                    map[search_column] = column_value
                else:
                    search_column = "meta_severity"
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
                re_match = re.compile(r'^\d{1,10}\.?\d{0,10}$')
                for k in map:
                    if (k in ["meta_age", "meta_sample_id2","meta_dataset","level2","meta_severity","meta_patient_id"]):
                        l = []
                        for ki in map[k]:
                            if re_match.findall(ki):
                                # if string only contains numbers, we need to convert it to integer to search
                                # age contains strings and integers, but the front-end will always process them to strings
                                if ki.isdigit():
                                    l.append(int(ki))
                                else:
                                    l.append(int(float(ki)))
                            else:
                                l.append(ki)
                        q = {k: {"$in": l}}
                        print(q)
                        construct.append(q)
                print(construct)
                print("debugging")
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
                        'sample_id': r['meta_sample_id2'],
                        'age': r['meta_age'],
                        'prediction': r['level2'],
                        'donor': r['meta_patient_id'],
                        'dataset': r['Arunachalam_2020'],
                        'status': r['meta_severity']
                    })

        response = {
                'draw': draw,
                'iTotalRecords': total_records,
                'iTotalDisplayRecords': total_records_filter,
                'aaData': data,
        }

        return jsonify(response)


def get_field(field_name):
    key = mongo.single_cell_meta.distinct(field_name)
    #uniq_field = mongo.single_cell_meta.aggregate([{"$group": {"_id": '$%s' % field_name}}]);
    #key = [r['_id'] for r in uniq_field]
    print("%s has %d uniq fields" % (field_name, len(key)))
    return key

