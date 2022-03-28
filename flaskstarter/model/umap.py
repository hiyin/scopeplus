# Data Model
from wtforms import form, fields
import flask_admin.contrib.pymongo as admin_py
from flask_admin.contrib.pymongo.filters import BooleanEqualFilter

class UmapForm(form.Form):
    id = fields.StringField('id')
    UMAP1 = fields.StringField('UMAP1')
    UMAP2 = fields.StringField('UMAP2')

# View

class UmapView(admin_py.ModelView):
    column_list = ('id', 'UMAP1', 'UMAP2')
    form = UmapForm  # Specifies Data Model



