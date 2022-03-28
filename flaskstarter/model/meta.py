from wtforms import form, fields
import flask_admin.contrib.pymongo as admin_py
from flask_admin.contrib.pymongo import filters

class MetaForm(form.Form):
    id = fields.StringField('id')
    sample_id = fields.StringField('sample_id')
    donor = fields.StringField('donor')
    age = fields.StringField('age')
    scClassify_prediction = fields.StringField('scClassify_prediction')
    status = fields.StringField('Status_on_day_collection_summary')
    dataset = fields.StringField('dataset')


# View
class MetaView(admin_py.ModelView):
    column_list = ('id', 'sample_id', 'donor', 'age', 'scClassify_prediction', 'Status_on_day_collection_summary', 'dataset')
    form = MetaForm  # Specifies Data Model

    column_sortable_list = ('age', 'donor')

    column_filters = (filters.FilterEqual('id', 'id'),
                      filters.FilterLike('age','age'),
                      filters.FilterEqual('age','age'),)

    column_searchable_list = ('id', 'sample_id')