# -*- coding: utf-8 -*-

from sqlalchemy import Column, Text, String

from ..extensions import db
from ..utils import get_current_time

from flask_login import current_user
from flask_admin.contrib import sqla


class SampleMetaAllModel(db.Model):

    __tablename__ = 'sample_meta_all'

    id = Column(db.Text, primary_key=True)

    sample_id = Column(db.Text)

    donor = Column(db.Text)

    age = Column(db.Text)

    Status_on_day_collection_summary = Column(db.Text)

    dataset = Column(db.Text)


# Customized MyTask model admin
class SampleMetaAllAdmin(sqla.ModelView):
    column_sortable_list = ('id', 'sample_id', 'Status_on_day_collection_summary')

    column_filters = ('id', 'sample_id', 'age')

    def __init__(self, session):
        super(SampleMetaAllAdmin, self).__init__(SampleMetaAllModel, session)

#    def is_accessible(self):
#        if current_user.role == 'admin':
#            return current_user.is_authenticated()
