# -*- coding: utf-8 -*-

from sqlalchemy import Column, Text, String

from ..extensions import db
from ..utils import get_current_time

from flask_login import current_user
from flask_admin.contrib import sqla


class covid2k_metaModel(db.Model):

    __tablename__ = 'covid2k_meta'



    sample_id = Column(db.Text, primary_key=True)

    id = Column(db.Text)

    X = Column(db.Text)

    donor = Column(db.Text)

    age = Column(db.Text)

    Status_on_day_collection_summary = Column(db.Text)

    dataset = Column(db.Text)


# Customized MyTask model admin
class covid2k_metaAdmin(sqla.ModelView):
    column_sortable_list = ('X', 'sample_id', 'Status_on_day_collection_summary')

    column_filters = ('X', 'sample_id', 'age')

    def __init__(self, session):
        super(covid2k_metaAdmin, self).__init__(covid2k_metaModel, session)

#    def is_accessible(self):
#        if current_user.role == 'admin':
#            return current_user.is_authenticated()