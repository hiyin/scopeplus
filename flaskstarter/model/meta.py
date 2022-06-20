from wtforms import form, fields
import flask_admin.contrib.pymongo as admin_py
from flask_admin.contrib.pymongo import filters

class MetaForm(form.Form):
    id = fields.StringField('id')
    meta_sample_id2 = fields.StringField('meta_sample_id2')
    meta_patient_id = fields.StringField('meta_patient_id')
    meta_age = fields.StringField('meta_age')
    level2 = fields.StringField('level2')
    meta_severity = fields.StringField('meta_severity')
    meta_dataset = fields.StringField('meta_dataset')


# View
class MetaView(admin_py.ModelView):
    column_list = ('id', 'meta_sample_id2', 'meta_patient_id', 'meta_age', 'level2', 'meta_severity', 'meta_dataset')
    form = MetaForm  # Specifies Data Model

    column_sortable_list = ('meta_age', 'meta_patient_id')

    column_filters = (filters.FilterEqual('id', 'id'),
                      filters.FilterLike('meta_age','meta_age'),
                      filters.FilterEqual('meta_age','meta_age'),)

    column_searchable_list = ('id', 'meta_sample_id2')