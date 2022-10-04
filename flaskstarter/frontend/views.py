# -*- coding: utf-8 -*-

from uuid import uuid4

from itsdangerous import URLSafeSerializer

from flask import (Blueprint, render_template, current_app, request,
                   flash, url_for, redirect)
from flask_login import (login_required, login_user, current_user,
                         logout_user, login_fresh)


from ..tasks import MyTaskForm
from ..user import Users, ACTIVE
from ..extensions import db, login_manager, mongo
from .forms import (SignupForm, LoginForm, RecoverPasswordForm,
                    ChangePasswordForm, ContactUsForm)
from .models import ContactUs
from ..utils import TMP_FOLDER
from ..emails import send_async_email
import sqlite3
from datetime import datetime
import os
import pandas as pd
from bson.son import SON
import pprint

frontend = Blueprint('frontend', __name__)

@frontend.route('/dashboard')
@login_required
def dashboard():

    _task_form = MyTaskForm()

    return render_template('dashboard/dashboard.html', task_form=_task_form, _active_dash=True)


def get_all_study_meta():
    res = list(mongo.pbmc_all_study_meta_v4.find())
    print(res[1])
    #uniq_field = mongo.single_cell_meta.aggregate([{"$group": {"_id": '$%s' % field_name}}]);
    #key = [r['_id'] for r in uniq_field]
    
    return res


def get_field_count():
    pipeline = [
        {"$unwind": "$level2"},
        {"$group": {"_id": "$level2", "count": {"$sum": 1}}},
        {"$project": SON([("count", 1), ("_id", 1)])}
    ]

    res = list(mongo.single_cell_meta.aggregate(pipeline))

    # for item in res:
    #     print(item)
    #     print(item['_id'])
    #     print(item["count"])
    return res
    
# Replace old get field count method    
def get_celltype_count():
    # db.createCollection('stats_celltype_count')
    # db.stats_celltype_count.insertMany(db.single_cell_meta_v4.aggregate([{"$group": {"_id": "$level2", "count":{"$sum": 1}}}]).toArray())
    collection_name = 'stats_celltype_count'
    res = list(mongo.stats_celltype_count.find())

    return res

def get_overview():
    res = list(mongo.stats_meta_overview.find())

    return res

# Load landing page data
filepath = os.path.abspath(os.getcwd())
filename = os.path.join(
    filepath, 'flaskstarter/PBMC_study_meta/PBMC_study_meta.csv')
df = pd.read_csv(filename)
df = df.dropna(subset=['Country Code'])

@frontend.route('/', methods=["GET", "POST"])
def index():
    data = df.to_dict()
    fdataset = get_all_study_meta()
    res = get_celltype_count()
    counting = get_overview()
    counts = counting
    #pprint.pprint(res)
    if current_user.is_authenticated:
        return render_template('tasks/landing.html',  data=data, _active_home=True, fdataset=fdataset, fcelltype=res, counting=counting, counts=counts)
    return render_template('tasks/landing.html',  data=data, _active_home=True, fdataset=fdataset, fcelltype=res, counting=counting, counts=counts)
# @frontend.route('/')
# def index():
#     # current_app.logger.debug('debug')
#     if current_user.is_authenticated:
#         return redirect(url_for('tasks.table_view'))

#     return render_template('tasks/landing.html', _active_home=True)

@frontend.route('/tutorial', methods=['GET', 'POST'])
def tutorial():
    return render_template('frontend/tutorial.html')

@frontend.route('/contact-us', methods=['GET', 'POST'])
def contact_us():
    form = ContactUsForm()

    if form.validate_on_submit():
        _contact = ContactUs()
        form.populate_obj(_contact)
        db.session.add(_contact)
        db.session.commit()

        flash('Thanks! We\'ll get back to you shortly!', 'success')

        return redirect(url_for('frontend.contact_us'))

    return render_template('frontend/contact_us.html', form=form)


@frontend.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('frontend.index'))

    form = LoginForm(login=request.args.get('login', None),
                     next=request.args.get('next', None))
    print("Debug login")   
    print(request)                 

    if form.validate_on_submit():
        user, authenticated = Users.authenticate(form.login.data, form.password.data)
        print(request.form)
        print(request.args)            

        if user and authenticated:

            if user.status_code != 2:
                flash("Please verify your email address to continue", "danger")
                return redirect(url_for('frontend.login'))

            remember = request.form.get('remember') == 'y'

            if login_user(user, remember=remember):
                flash("Logged in", 'success')
            return redirect(form.next.data or url_for('frontend.index'))
        else:
            flash('Sorry, invalid login', 'danger')

    return render_template('frontend/login.html', form=form, _active_login=True)


@frontend.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out', 'success')
    return redirect(url_for('frontend.index'))


@frontend.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('frontend.dashboard'))

    form = SignupForm(next=request.args.get('next'))

    if form.validate_on_submit():
        user = Users()
        user.status_code = 2
        user.account_type = 0
        form.populate_obj(user)

        db.session.add(user)
        db.session.commit()

        confirm_user_mail(form.name.data, form.email.data)

        flash(u'Registration successful! Please verify!', 'success')
        return redirect(url_for('frontend.login'))

    return render_template('frontend/signup.html', form=form,
                           _active_signup=True)


def confirm_user_mail(name, email):
    s = URLSafeSerializer('serliaizer_code')
    key = s.dumps([name, email])

    subject = 'Confirm your account for ' + current_app.config['PROJECT_NAME']
    url = url_for('frontend.confirm_account', secretstring=key, _external=True)
    html = render_template('macros/_confirm_account.html',
                           project=current_app.config['PROJECT_NAME'],
                           url=url)

    send_async_email(subject, html, email)


@frontend.route('/confirm_account/<secretstring>', methods=['GET', 'POST'])
def confirm_account(secretstring):
    s = URLSafeSerializer('serliaizer_code')
    uname, uemail = s.loads(secretstring)
    user = Users.query.filter_by(name=uname).first()
    user.status_code = ACTIVE
    db.session.add(user)
    db.session.commit()
    flash(u'Your account was confirmed succsessfully!!!', 'success')
    return redirect(url_for('frontend.login'))


@frontend.route('/change_password', methods=['GET', 'POST'])
def change_password():

    if current_user.is_authenticated:
        if not login_fresh():
            return login_manager.needs_refresh()

    form = ChangePasswordForm(email_activation_key=request.values["email_activation_key"],
                              email=request.values["email"])

    if form.validate_on_submit():
        update_password(form.email.data, form.email_activation_key.data, form.password.data)
        flash(u"Your password has been changed, log in again", "success")
        return redirect(url_for("frontend.login"))

    return render_template("frontend/change_password.html", form=form)


def update_password(email, email_activation_key, password):
    user = Users.query.filter_by(email_activation_key=email_activation_key, email=email).first()
    user.password = password
    user.email_activation_key = None
    db.session.add(user)
    db.session.commit()


@frontend.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    form = RecoverPasswordForm()

    if form.validate_on_submit():
        user = Users.query.filter_by(email=form.email.data).first()

        if user:
            flash('Please see your email for instructions on how to access your account', 'success')

            user.email_activation_key = str(uuid4())
            db.session.add(user)
            db.session.commit()

            subject = 'Reset your password in ' + current_app.config['PROJECT_NAME']
            url = url_for('frontend.change_password', email=user.email,
                          email_activation_key=user.email_activation_key, _external=True)
            html = render_template('macros/_reset_password.html', project=current_app.config['PROJECT_NAME'],
                                   name=user.name, url=url)

            send_async_email(subject, html, user.email)

            return render_template('frontend/reset_password.html', form=form)
        else:
            flash('Sorry, no user found for that email address', 'danger')

    return render_template('frontend/reset_password.html', form=form)


@frontend.route('/terms')
def terms():
    return "To be updated soon.."


@frontend.route('/about-us')
def about_us():
    return "To be updated soon.."
