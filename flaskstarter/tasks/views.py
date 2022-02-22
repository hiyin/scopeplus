# -*- coding: utf-8 -*-

from flask import Blueprint, render_template, flash, redirect, url_for, send_file, request
from flask_login import login_required, current_user

from ..extensions import db

from .forms import MyTaskForm
from .models import MyTaskModel
from flaskstarter.covid2k_meta import covid2k_metaModel
import csv

import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import json
tasks = Blueprint('tasks', __name__, url_prefix='/tasks')

@tasks.route('/my_tasks', methods=['GET', 'POST'])
@login_required
def my_tasks(condition=''):
    print('debug...')
    print('change condition...')
    print(condition)
    graphJSON = {}

    if not condition:
        _all_tasks = covid2k_metaModel.query.all()
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
    col_values = get_col_values()


    return render_template('tasks/my_tasks.html',
                           all_tasks=_all_tasks,
                           col_values=col_values,
                           graphJSON=graphJSON,
                           _active_tasks=True)


@tasks.route('/home', methods=['GET', 'POST'])
def home(condition=''):
    print('debug...')
    print('change condition...')
    print(condition)
    graphJSON = {}

    if not condition:
        _all_tasks = covid2k_metaModel.query.all()
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
    col_values = get_col_values()

    return render_template('tasks/landing.html',
                           all_tasks=_all_tasks,
                           col_values=col_values,
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


@tasks.route('/view_task/<id>', methods=['GET', 'POST'])
@login_required
def view_task(id):
    _task = MyTaskModel.query.filter_by(id=id, users_id=current_user.id).first()

    if not _task:
        flash('Oops! Something went wrong!.', 'danger')
        return redirect(url_for("tasks.my_tasks"))

    return render_template('tasks/view_task.html',
                           task=_task)


@tasks.route('/add_task', methods=['GET', 'POST'])
@login_required
def add_task():

    _task = MyTaskModel()

    _form = MyTaskForm()

    if _form.validate_on_submit():

        _task.users_id = current_user.id

        _form.populate_obj(_task)

        db.session.add(_task)
        db.session.commit()

        db.session.refresh(_task)
        flash('Your task is added successfully!', 'success')
        return redirect(url_for("tasks.my_tasks"))

    return render_template('tasks/add_task.html', form=_form, _active_tasks=True)


@tasks.route('/delete_task/<id>', methods=['GET', 'POST'])
@login_required
def delete_task(id):
    _task = MyTaskModel.query.filter_by(id=id, users_id=current_user.id).first()

    if not _task:
        flash('Oops! Something went wrong!.', 'danger')
        return redirect(url_for("tasks.my_tasks"))

    db.session.delete(_task)
    db.session.commit()

    flash('Your task is deleted successfully!', 'success')
    return redirect(url_for('tasks.my_tasks'))


@tasks.route('/edit_task/<id>', methods=['GET', 'POST'])
@login_required
def edit_task(id):
    _task = MyTaskModel.query.filter_by(id=id, users_id=current_user.id).first()

    if not _task:
        flash('Oops! Something went wrong!.', 'danger')
        return redirect(url_for("tasks.my_tasks"))

    _form = MyTaskForm(obj=_task)

    if _form.validate_on_submit():

        _task.users_id = current_user.id
        _form.populate_obj(_task)

        db.session.add(_task)
        db.session.commit()

        flash('Your task updated successfully!', 'success')
        return redirect(url_for("tasks.my_tasks"))

    return render_template('tasks/edit_task.html', form=_form, task=_task, _active_tasks=True)
