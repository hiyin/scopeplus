# -*- coding: utf-8 -*-

from sqlalchemy import Column, Text, String

from ..extensions import db
from ..utils import get_current_time

from flask_login import current_user
from flask_admin.contrib import sqla
import sqlite3

connection = sqlite3.connect('/tmp/flaskstarter-instance/db.sqlite', check_same_thread=False)
cursor = connection.execute('select * from covid2k_dense_2k')


class covid2k_dense2kModel(db.Model):

    __tablename__ = 'covid2k_dense_2k'

    gene_id = Column(db.Text, primary_key=True)
    for column in cursor.description:
        if 'gene_id' not in column[0]:
            locals()[column[0]] = Column(db.Text)


# Customized MyTask model admin
class covid2k_dense2kAdmin(sqla.ModelView):
    column_sortable_list = [description[0] for description in cursor.description]

    column_filters = [description[0] for description in cursor.description]

    def __init__(self, session):
        super(covid2k_dense2kAdmin, self).__init__(covid2k_dense2kModel, session)

#    def is_accessible(self):
#        if current_user.role == 'admin':
#            return current_user.is_authenticated()