# -*- coding: utf-8 -*-

from sqlalchemy import Column, Text, String

from ..extensions import db
from ..utils import get_current_time

from flask_login import current_user
from flask_admin.contrib import sqla
import sqlite3

connection = sqlite3.connect('/tmp/flaskstarter-instance/db.sqlite')
cursor = connection.execute('select * from covid2k_dense_2k')


class covid2k_denseModel(db.Model):

    __tablename__ = 'covid2k_dense'

    gene_id = Column(db.Text, primary_key=True)

    AAACCTGAGAAACCTA_MH9179824 = Column(db.Text)


# Customized MyTask model admin
class covid2k_denseAdmin(sqla.ModelView):
    column_sortable_list = ('gene_id', 'AAACCTGAGAAACCTA_MH9179824' )

    column_filters = ('gene_id', 'AAACCTGAGAAACCTA_MH9179824')

    def __init__(self, session):
        super(covid2k_denseAdmin, self).__init__(covid2k_denseModel, session)

#    def is_accessible(self):
#        if current_user.role == 'admin':
#            return current_user.is_authenticated()