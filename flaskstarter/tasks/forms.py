# -*- coding: utf-8 -*-

from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField, SelectField
from wtforms.validators import Required, Length

class MyTaskForm(FlaskForm):
    task = TextAreaField(u'Your Task', [Required(), Length(5, 2048)])
    submit = SubmitField(u'Save Task')

class UmapForm(FlaskForm):
    def __init__(self,choices=[('def','Default')]):
        self.cell_color = SelectField('cell_color',choices=choices)