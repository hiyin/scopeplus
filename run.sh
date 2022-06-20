#!/bin/bash
baseDir=/home/d24h_prog5/covid19_cell_atlas_portal
cd $baseDir
source venv/bin/activate
export FLASK_ENV=development
export FLASK_APP=manage.py
nohup python3 -m flask run --host='0.0.0.0' > /home/d24h_prog5/flasklog/log.txt 2>&1 &
