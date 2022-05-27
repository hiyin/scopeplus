# -*- coding: utf-8 -*-

from ast import If
from operator import index
from flask import Blueprint, render_template, flash, redirect, url_for, send_file, request, jsonify
from flask_login import login_required, current_user
from ..extensions import db, mongo,scfeature

from flaskstarter.covid2k_meta import covid2k_metaModel
from flaskstarter.tasks.forms import UmapForm
from plotly.subplots import make_subplots
import csv
import gzip
import zipfile
import shutil
import re
import glob
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
from os.path import exists,basename
import dash_bio
from sklearn import preprocessing
import seaborn as sns


tasks = Blueprint('tasks', __name__, url_prefix='/tasks')

# sub folders to manage different flask instances: not a good place to put, should be in API endpoint?
user_timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
#user_timestamp = "test"
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

    # Get params from html
    cell_color = request.form.get('name_opt_col')
    cell_gene = request.form.get('name_opt_gene')
    df_genes = pd.read_csv(TMP_FOLDER+"/genes.tsv", sep="\t", header=None)

    # Add a flag to check if the local folder cached the ids
    flag_idmissing = 0    
    # Search box is empty
    if(len(collection_searched)==0):
        # No ids.csv file is presented:
        if(not (exists(user_tmp[-1] + '/ids.csv'))):
            shutil.copy2(TMP_FOLDER+'/default/umap.csv', user_tmp[-1] + '/umap.csv') # complete target filename given
            shutil.copy2(TMP_FOLDER+'/default/meta.tsv', user_tmp[-1] + '/meta.tsv') # complete target filename given
            # Change to show default case
            flag_idmissing = 1

        # ID is presented, no meta data:
        elif(not (exists(user_tmp[-1] + '/meta.tsv'))):
            f = user_tmp[-1] + "/ids.csv"
            _byid = pd.read_csv(f).values.tolist()
            lookups = list(np.squeeze(_byid))
            meta = mongo.single_cell_meta.find({'id': {'$in': lookups}})
            write_file_meta(user_tmp[-1],meta)
        
        # If ID is presented, no umap:
        elif(not (exists(user_tmp[-1] + '/umap.tsv'))):
            f = user_tmp[-1] + "/ids.csv"
            _byid = pd.read_csv(f).values.tolist()
            lookups = list(np.squeeze(_byid))
            umap = mongo.umap.find({'id': {'$in': lookups}})
            write_umap(user_tmp[-1], umap)

    # If search box not empty, write id meta umap
    else:
        meta = mongo.single_cell_meta.find({'_id': {'$in': collection_searched}})
        ids = write_id_meta(user_tmp[-1], meta)
        umap = mongo.umap.find({'id': {'$in': ids}})
        write_umap(user_tmp[-1], umap)

        # Assume ids,meta files are provided

        # No cell color is provided
        if(cell_color is None):
            cell_color="level2"
        if(cell_gene is None):
            print("----No gene selected")
        else:
            print("----Gene is selected")

            # The gene name selected is avaliable
            if(cell_gene in df_genes.values):
                # Lookup gene in default dir or user dir based on condition
                flag_same_query =  is_same_query(user_tmp[-1]+"/meta.tsv",collection_searched)
                # Gene table is not cached or the query is differen from the previous
                if((not exists(user_tmp[-1] + '/'+cell_gene+'.tsv')) or 
                (flag_same_query == False)):

                    # Lookup from the database
                    if(flag_idmissing==0):
                        _byid = pd.read_csv(user_tmp[-1]  + "/ids.csv").values.tolist()
                    else:
                        _byid = pd.read_csv(TMP_FOLDER+'/default/ids.csv').values.tolist()
                        
                    lookups = list(np.squeeze(_byid))
                    print("Query ing")
                    checkpoint_time = time.time()
                    query = mongo.matrix.find({'barcode': {'$in': lookups},'gene_name':cell_gene})
                    print("query finished --- %s seconds ---" % (time.time() - checkpoint_time))

                    ##text=List of strings to be written to file
                    # Force to write to the user folder
                    with open(user_tmp[-1] + '/'+cell_gene+'.tsv', 'w') as file:
                        file.write("\t".join(["_id","gene","barcode",cell_gene]))
                        file.write('\n')
                        for line in query:
                            file.write("\t".join([str(e) for e in line.values()]))
                            file.write('\n')

                    checkpoint_time = time.time()
                    print("write finished --- %s seconds ---" % (time.time() - checkpoint_time))
            else:
                cell_gene = None
                print("Gene not found")
        # Plot umap
        checkpoint_time = time.time()
        graphJSON,graphJSON2,df_plot = plot_umap(cell_color,cell_gene)
        print("Plot finished --- %s seconds ---" % (time.time() - checkpoint_time))

        # Pass color options to the html
        colors = list(df_plot.columns.values.ravel())
        genes = df_genes.values.ravel()
    return render_template('tasks/show_plot.html', graphJSON=graphJSON,graphJSON2=graphJSON2,colors=colors,genes=genes)

@tasks.route('/show_scfeature', methods=['GET', 'POST'])
def show_scfeature():     

    # Get params from html
    
    
    dataset = request.form.get('name_opt_dataset')
    cell_type = request.form.get('name_opt_celltype')
    feature = request.form.get('name_opt_feature')


    print("Html params",dataset,cell_type,feature)	

    if(dataset == None):
        dataset = "Arunachalam_2020"

    fileds_dataset = mongo.single_cell_meta.distinct("meta_dataset")
    fileds_celltypes = get_field("level2")
    fileds_dataset_2 = mongo.single_cell_meta.aggregate(
            [
                {"$match":{"meta_dataset":dataset}},
                {"$group": {"_id": {"meta_dataset": "$meta_dataset", "meta_sample_id2": "$meta_sample_id2"}}}
            ]

    )
    mata_sample_id2 = [x["_id"]["meta_sample_id2"] for x in list(fileds_dataset_2)]
    print("Sample id 2: ",mata_sample_id2)

    datasets = fileds_dataset
    celltypes = ["All"]
    celltypes=celltypes+fileds_celltypes
    features = []
    
    try:
        # Query propotion
        propotion = scfeature.proportion_raw.find({'meta_dataset': {'$in': mata_sample_id2}})
        df_propotion = pd.DataFrame(list(propotion))
        df_propotion.to_csv(user_tmp[-1]+"/df_proportion_raw_"+dataset+".csv")
        df_d = df_propotion.drop(columns=["_id","meta_scfeature_id"])
        df_melt=df_d.melt(id_vars=['meta_dataset','meta_severity'],value_name="propotion",var_name="cell_type")
        #df_melt.to_csv(user_tmp[-1]+"/proportion_melt.tsv", sep="\t")
        fig = px.bar(df_melt, x="meta_dataset", y="propotion", color="cell_type",facet_col = "meta_severity",title="Overview: cell type proportion in " + dataset,
        color_discrete_sequence=sns.color_palette("tab20").as_hex())
        fig.update_xaxes(matches=None)
        fig.update_layout(
            autosize=True, width=1200, height=600
        )
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        graphJSON = None
        print("Error generating figure propotion",e)
    
    if(cell_type == "All"):	
        graphJSON2 = None
    else:
        try:
            # Query dendrogram for each gene
            gene_prop_celltype = scfeature.gene_prop_celltype.find({'meta_dataset': {'$in': mata_sample_id2}})
            df_gene_prop_celltype = pd.DataFrame(list(gene_prop_celltype))
            df_gene_prop_celltype.to_csv(user_tmp[-1]+"/df_gene_prop_celltype_"+dataset+".csv")
            df_gene_prop_celltype = df_gene_prop_celltype.drop(columns=["_id"])
            if(not(cell_type in celltypes)):
                graphJSON2 = None
            else:
                select_type = cell_type
                fig2 = process_dendrogram(df_gene_prop_celltype,select_type)
                graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            graphJSON2 = None
            print("Error generating figure dendrogram",e)

    try:
        # Query boxplot
        pathway_mean = scfeature.pathway_mean.find({'meta_dataset': {'$in': mata_sample_id2}})
        df_pathway_mean = pd.DataFrame(list(pathway_mean))
        df_pathway_mean.to_csv(user_tmp[-1]+"/df_pathway_mean_"+dataset+".csv")
        df_pathway_mean = df_pathway_mean.drop(columns=["_id"])
        fig3,features = process_boxplot(df_pathway_mean,cell_type,plot_type="pathway",feature=feature)

        if(feature is None):
            graphJSON3 = None
        elif(not(feature in features)):
            graphJSON3 = None
        else:
            graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
        if(cell_type == "All"):	
            graphJSON4 = None
        elif(not(cell_type in celltypes)):
            graphJSON4 = None
        else:
            select_type = cell_type
            #df_pathway_mean.to_csv(user_tmp[-1]+"/df_test"+select_type+".csv")
            fig4 = process_dendrogram(df_pathway_mean,select_type,plot_type="pathway")
            graphJSON4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    except Exception as e:
        graphJSON3 = None
        graphJSON4 = None
        print("Error generating figure boxplot",e)


    return render_template('tasks/show_scfeature.html', graphJSON=graphJSON,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4,datasets=datasets,celltypes=celltypes,features=features)

## Add by junyi
def process_dendrogram(data,cell_type,plot_type="gene"):
    data= data.set_index("meta_dataset")

    if(plot_type=="gene"):
        celltype = data.columns.values
        celltype = np.array([x.split('--')[0] for x in celltype] )
    else:
        data = data.drop(columns=["meta_scfeature_id"])
        celltype = data.columns[2:].values
        celltype = np.array([x.split('--')[1] for x in celltype] )

    data_patient = data.index.values 
    condition = data.meta_severity.values 

    if(cell_type in celltype):
        data = data.iloc[:, np.where(celltype == cell_type)[0]] 
    else:
        return None

    condition_colour =['#f00314', '#ff8019',   
                                '#3bb5ff', '#0500c7' , '#5c03fa', '#de00ed' , '#fae603']
    remove =  np.where( np.array( data.sum(axis=0)  ) == 0)[0]
    if remove.size > 0:
        data.drop(data.columns[remove],axis=1,inplace=True) 
    data_np = np.array( data )
    top_feature = np.var(data_np, axis = 0) 
    top_feature = np.argsort(top_feature)
    top_feature = top_feature[::-1][0:50]

    data = data.iloc[:,top_feature ]

    data["condition"] = condition
    my_palette = dict(zip( set(condition ) , condition_colour[0:len(set(condition))]))
    col_colors = data.condition.map(my_palette)

    data = data.drop('condition', 1)
    data = data.transpose()

    x = data.transpose().values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df = df.transpose()
    df.columns = data.columns
    df.index = data.index

    if(plot_type!="gene"):
        df = df.transpose()


    fig2 = dash_bio.Clustergram(
        data=df,
        column_labels=list(df.columns.values),
        row_labels=list(df.index),
        #column_colors= list(col_colors),
        row_colors_label= "Condition",
        color_map= [
        [0.0, '#000080'],
        [0.5, '#ffffff'],
        [1.0, '#ff0000']
        ],
        height=1000,
        width=1200
    )
    fig2.update(layout_showlegend=False)    
    return fig2

def process_boxplot(input_data,cell_type,plot_type="gene",feature=None):


    data = input_data.set_index("meta_scfeature_id")
    data = data.drop(columns=['meta_dataset','meta_severity'])

    columns = data.columns.values
    
    if(plot_type=="gene"):
        celltypes = np.array([x.split('--')[0] for x in columns] )
        # if(feature is None):
        #     feature = "KLF6"
        features = list(set(np.array([x.split('--')[1] for x in columns] )))
    else:
        celltypes = np.array([x.split('--')[1] for x in columns] )
        # if(feature is None):
        #     feature = "HALLMARK-ADIPOGENESIS"            
        features = list(set(np.array([x.split('--')[0] for x in columns] )))
    if(not(feature in features)):
        feature = features[0]           

    if(not(cell_type is None)):
        if(not(cell_type == "All")):
            data = data.iloc[:, np.where(celltypes == cell_type)[0]] 

    data_patient = data.index.values 
    data["patient"] = [x.split('_cond_')[0] for x in data_patient] 
    data["condition"] = [x.split('_cond_')[1] for x in data_patient] 
  
    data=data.melt(id_vars=['patient','condition'])
    data = data[data["variable"].str.contains(feature) ]

    fig = px.box(data,  x="variable", y="value",color="condition")
    fig.update_layout(
            autosize=False, width=1200, height=800,
            # legend=dict(
            #         title=None, orientation="h", y=0, yanchor="top", x=0.5, xanchor="center"
            #     )
            )

    return fig, features

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

def plot_stack_bar(df):
    l = []
    fig = px.histogram(df, x="sex", y="total_bill",
                color='smoker', barmode='group',
                height=400)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

## Create by junyi
def plot_umap(cell_color='scClassify_prediction',gene_color=None):
    df = pd.read_csv(user_tmp[-1] + '/umap.csv', index_col=0)
    df.columns = ["umap_0","umap_1"]
    df_meta = pd.read_csv(user_tmp[-1] + '/meta.tsv', index_col=1,sep="\t")
    df_plot = df.merge(df_meta, left_index=True, right_index=True)


    if(not(gene_color is None)):
        df_gene = pd.read_csv(user_tmp[-1] + '/'+gene_color+'.tsv',sep="\t", index_col=2)
        df_plot = df_plot.merge(df_gene, left_index=True, right_index=True,how="left")
        print(df_plot.shape)
        df_plot[gene_color] = df_plot[gene_color].fillna(value=0)
        #df_plot[gene_color] = df_plot[gene_color].fillna(value=0)
        #fig2 = px.violin(df_plot, y=gene_color, x=cell_color, box=False, points=False)
        #df_plot[gene_color] = np.clip(df_plot[gene_color],np.percentile(df_plot[gene_color], 5),np.percentile(df_plot[gene_color], 95))
        fig2 = px.box(df_plot, y=gene_color, x=cell_color,color=cell_color,color_discrete_sequence=
            ["#F8A19F","#8E321C","#F6222E","#F1CE63","#B6992D","#59A14F","#499894","#4E79A7",
            "#A6AAFE","#5930FB","#500EE2","#7427B9","#931ADD","#A04DB9","#C585AF","#B8418A","#AD0267","#7A325C","#cccccc","#b2b2b2"])

        fig2.update_layout(
        autosize=False, width=900, height=600)
        graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        fig = px.scatter(
            df_plot, x="umap_0", y="umap_1",color=gene_color,color_continuous_scale="Viridis")
        fig.update_layout(
                autosize=False, width=900, height=600)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    else:
        fig = px.scatter(
            df_plot, x="umap_0", y="umap_1",color=cell_color,color_discrete_sequence=
            ["#F8A19F","#8E321C","#F6222E","#F1CE63","#B6992D","#59A14F","#499894","#4E79A7",
            "#A6AAFE","#5930FB","#500EE2","#7427B9","#931ADD","#A04DB9","#C585AF","#B8418A","#AD0267","#7A325C","#cccccc","#b2b2b2"]
            ,color_continuous_scale="Viridis")
        fig.update_layout(
                autosize=False, width=900, height=600)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        fig2 = None
        graphJSON2 = None

    return graphJSON,graphJSON2,df_plot


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
    fsampleid = get_field("meta_sample_id2")
    fage = get_field("meta_age")
    print(fage)
    fdonor = get_field("meta_patient_id")
    fprediction = get_field("level2")
    fstatus = get_field("meta_severity")
    fdataset = get_field("meta_dataset")
    return render_template('tasks/table_view.html',
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

# Write id meta file
def write_id_meta(path, towrite):
    fid = path + '/ids.csv'
    fmeta = path + '/meta.tsv'

    print('writing ids and meta')
    ids = []
    with open(fid, 'w') as file_id,open(fmeta, 'w') as file_meta:
        file_meta.write("\t".join([str(e) for e in towrite[0].keys()]))
        file_meta.write('\n')

        for r in towrite:
            # Writ id
            row = str(r['id'])
            file_id.write(row)
            file_id.write('\n')
            ids.append(row)
            # Writ meta
            file_meta.write("\t".join([str(e) for e in r.values()]))
            file_meta.write('\n')
    return ids

# Write file
def write_file_meta(path, towrite):
    fn = path + '/meta.tsv'
    print('writing meta to' + fn)

    ##text=List of strings to be written to file
    #print(towrite[0])
    with open(fn, 'w') as file:
        file.write("\t".join([str(e) for e in towrite[0].keys()]))
        file.write('\n')

        for line in towrite:
            file.write("\t".join([str(e) for e in line.values()]))
            file.write('\n')

def write_umap(path, towrite):
    fn = path + '/umap.csv'
    print('writing umap to' + fn)
    with open(fn, 'w') as file:
        for r in towrite:
            file.write(",".join([str(r['id']), str(r['UMAP1']), str(r['UMAP2'])]))
            file.write('\n')
    return fn

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

def is_same_query(meta_path,collection_searched):
    try:
        df_meta = pd.read_csv(meta_path, index_col=1,sep="\t")
        _ids = list(df_meta._id.values)
        _ids.sort()
        oid = [str(_id) for _id in collection_searched]
        oid.sort()
    except:
        print("Error when comparing query, return false")
        return False    

    return _ids == oid

def remove_files(path):
    for f in [
        path + '/ids.csv',
        path + '/matrix.zip',
        path + '/matrix.mtx.gz',
        path + '/genes.tsv.gz',
        path + '/barcodes.tsv.gz',
        path + '/meta.tsv']:
        if((exists(f))):
            shutil.os.remove(f)

# Download big file
@tasks.route('/download_meta',methods=['POST'])
def download_meta():
    meta = list(mongo.single_cell_meta.find({'_id': {'$in': collection_searched}}))
    print('writing ids to csv file only once, firstly load the data')
    write_file_byid(user_tmp[-1], meta)
    write_file_meta(user_tmp[-1], meta)
    #return send_file(user_tmp[-1] + '/ids.csv', as_attachment=True)
    return send_file(user_tmp[-1] + '/meta.tsv', as_attachment=True)

# Download big file
@tasks.route('/download_scfeature',methods=['POST'])
def download_scfeature():

    
    # list_files = [
    #     user_tmp[-1]+"/df_gene_prop_celltype.csv",
    #     user_tmp[-1]+"/df_pathway_mean.csv",
    #     user_tmp[-1]+"/df_proportion_raw.csv"
    # ]
    list_files = glob.glob( user_tmp[-1]+"/*_gene_prop_celltype*.csv")+\
    glob.glob( user_tmp[-1]+"/*_pathway_mean*.csv")+\
    glob.glob( user_tmp[-1]+"/*_proportion_raw*.csv")

    if(not (exists(user_tmp[-1] + '/scfeature.zip'))):
        with zipfile.ZipFile(user_tmp[-1] + '/scfeature.zip', 'w') as zipMe:        
            for file in list_files:
                if(exists(file)):
                    zipMe.write(file,arcname=basename(file), compress_type=zipfile.ZIP_DEFLATED)

    return send_file(user_tmp[-1] + '/scfeature.zip', as_attachment=True)



@tasks.route('/download_umap', methods=['POST'])
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

@tasks.route('/download_matrix',methods=['POST'])
def download_matrix():
    # No query is search, then we use default use case:
    if(len(collection_searched)==0):
        if(not (exists(user_tmp[-1] + '/ids.csv'))):
            return send_file(TMP_FOLDER+'/default/matrix.zip', as_attachment=True)
    else:
        # If query is presented, remove the result old queries
        if((exists(user_tmp[-1] + '/meta.tsv'))):
            flag_same_query = is_same_query(user_tmp[-1] + '/meta.tsv',collection_searched=collection_searched)
            # Different from the old query
            if(flag_same_query == False):
                print("Remove files because the old query is deprecared")
                remove_files(user_tmp[-1])
            else:
                print("Keep files because the old query is the same")
        # No old queries
        else:
            remove_files(user_tmp[-1])

        meta = mongo.single_cell_meta.find({'_id': {'$in': collection_searched}})
        ids = write_id_meta(user_tmp[-1], meta)

    # Down load 10x matrix if not exist
    if(not (exists(user_tmp[-1] + '/matrix.mtx.gz'))):
        _byid = pd.read_csv(user_tmp[-1] + "/ids.csv").values.tolist()
        lookups = list(np.squeeze(_byid))
        print("debugging")
        start_time2 = time.time()
        mtx = mongo.matrix.find({'barcode': {'$in': lookups}})
        print("query finished --- %s seconds ---" % (time.time() - start_time2))

        start_time2 = time.time()
        #query_counts = mtx.size()
        #mtx = list(mtx)
        doc_count = mongo.matrix.count_documents({'barcode': {'$in': lookups}})

        print("list finished --- %s seconds ---" % (time.time() - start_time2))

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
        write_10x_mtx(user_tmp[-1], dict_gene, dict_barcode, doc_count, mtx)

    if(not (exists(user_tmp[-1] + '/matrix.zip'))):
        list_files = [
            user_tmp[-1] + '/matrix.mtx.gz',
            user_tmp[-1] + '/genes.tsv.gz',
            user_tmp[-1] + '/barcodes.tsv.gz',
            user_tmp[-1] + '/meta.tsv'
        ]
        with zipfile.ZipFile(user_tmp[-1] + '/matrix.zip', 'w') as zipMe:        
            for file in list_files:
                if(exists(file)):
                    zipMe.write(file,arcname=basename(file), compress_type=zipfile.ZIP_DEFLATED)

    return send_file(user_tmp[-1] + '/matrix.zip', as_attachment=True)


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
                    search_column = "meta_sample_id2"
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
                        'dataset': r['meta_dataset'],
                        'status': r['meta_severity']
                    })

        response = {
                'draw': draw,
                'iTotalRecords': total_records,
                'iTotalDisplayRecords': total_records_filter,
                'aaData': data,
        }

        return jsonify(response)


@tasks.route('/home', methods=['GET', 'POST'])
@login_required
def home(condition=''):
    print('debug...')
    print('change condition...')
    print(condition)
    graphJSON = {}

    if not condition:
        _all_tasks = covid2k_metaModel.query.all()
        print(_all_tasks)
        print('show default')

    else:
        if isinstance(condition, str):
            if condition == 'all':
                _all_tasks = covid2k_metaModel.query.all()
                print(_all_tasks)
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
    col_values = get_col_values()

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
    return home('all')


@tasks.route('/age52',methods=['POST'])
def age_filter():
    return home('20')

@tasks.route('/download',methods=['POST'])
def download():
    plot()
    return send_file('/tmp/flaskstarter-instance/result.csv', as_attachment=True)


def get_col_values():
    #_col_values = covid2k_metaModel.query.with_entities(covid2k_metaModel.donor)
    donors = [c.donor for c in covid2k_metaModel.query.with_entities(covid2k_metaModel.donor).distinct()]
    print(len(donors))
    return donors



@tasks.route('/get_multiselect', methods=['POST'])
# uses list to store returned condition for my_tasks
def get_multiselect():
    selected_vals = request.form.getlist('multiselect')
    print(request.form)
    print(selected_vals)

    return home(selected_vals)


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

    #query = { 'donor': {'$in': selected_vals}}
    #ids = [x["_id"] for x in list(mongo.single_cell_meta.find(query, {"_id": 1}))]



def get_field(field_name):
    key = mongo.single_cell_meta.distinct(field_name)
    #uniq_field = mongo.single_cell_meta.aggregate([{"$group": {"_id": '$%s' % field_name}}]);
    #key = [r['_id'] for r in uniq_field]
    print("%s has %d uniq fields" % (field_name, len(key)))
    return key

