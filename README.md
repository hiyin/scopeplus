# Covid-19 Cell Atlas

Covid-19 Cell Atlas Portal is a Flask web application development based on flaskstarter boilerplate. It has all the ready-to-use bare minimum essentials.


## Demo
Video: https://d24h-my.sharepoint.com/:v:/g/personal/junyichen_d24h_hk/EZz9CeDLpq1Hl2snWY2f-dsBIP4Fq1Dx9hR4WRAWuzm22A?e=awW1nz


## Table of Contents
1. [Deployment](#deployment)
1. [Getting Started](#getting-started)
1. [Project Structure](#project-structure)
1. [Modules](#modules)
1. [Testing](#testing)
1. [Features](#features)
1. [Need Help?](#need-help)


## Deployment
start D24H server at port 5000: execute "source run.sh"
Below is run.sh snippet 
```bash
#!/bin/bash
source venv/bin/activate
export FLASK_ENV=development
export FLASK_APP=manage.py
nohup python3 -m flask run --host='0.0.0.0' > /home/d24h_prog5/flasklog/log.txt 2>&1 &
```

## Getting Started

clone the project

```bash
$ git clone https://github.com/ksh7/flask-starter.git
$ cd flask-starter
```

create virtual environment using python3 and activate it (keep it outside our project directory)

```bash
$ python3 -m venv /path/to/your/virtual/environment
$ source <path/to/venv>/bin/activate
```

For MAC/Linux local development
```bash
$ cd ../../
$ virtualenv venv
$ source venv/bin/activate
$ cd covid19_cell_atlas_portal
```

install dependencies in virtualenv

```bash
$ pip install -r requirements.txt
```

setup `flask` command for our app

```bash
$ export FLASK_APP=manage.py
$ export FLASK_ENV=development
```

create instance folder in `/tmp` directory (sqlite database, temp files stay here)

```bash
$ mkdir /tmp/flaskstarter-instance
```

initialize database and get two default users (admin & demo), check `manage.py` for details

```bash
$ flask initdb
```

start test server at `localhost:5000`

```bash
$ flask run
```


## Project Structure

```bash
d24h_prog5@d24hp5ubuntu:~/covid19_cell_atlas_portal$ tree -I "venv|__pycache__"
.
├── flaskstarter
│   ├── app.py
│   ├── config.py
│   ├── covid2k_dense
│   │   ├── __init__.py
│   │   └── models.py
│   ├── covid2k_meta
│   │   ├── __init__.py
│   │   └── models.py
│   ├── decorators.py
│   ├── emails
│   │   └── __init__.py
│   ├── extensions.py
│   ├── frontend
│   │   ├── forms.py
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── views.py
│   ├── __init__.py
│   ├── model
│   │   ├── meta.py
│   │   └── umap.py
│   ├── naso_tableview
│   │   ├── __init__.py
│   │   └── views.py
│   ├── sample_meta_all
│   │   ├── __init__.py
│   │   └── models.py
│   ├── settings
│   │   ├── forms.py
│   │   ├── __init__.py
│   │   └── views.py
│   ├── static
│   │   ├── bootstrap.bundle.min.js
│   │   ├── bootstrap.min.css
│   │   ├── chevron-down.png
│   │   ├── css
│   │   │   └── template.css
│   │   ├── D24H_Logo.png
│   │   ├── Dataset_Banner.jpeg
│   │   ├── download-file 1.svg
│   │   ├── explore.png
│   │   ├── folder-2.png
│   │   ├── folder.png
│   │   ├── Group 36.png
│   │   ├── Group 37.png
│   │   ├── Group 38.png
│   │   ├── Group.png
│   │   ├── jquery-3.6.0.min.js
│   │   ├── jquery.slim.min.js
│   │   ├── Login_Banner.jpeg
│   │   ├── Mask_Group.png
│   │   ├── microscope.png
│   │   ├── newspaper.png
│   │   └── Pointer.png
│   ├── tasks
│   │   ├── forms.py
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── views.py
│   ├── templates
│   │   ├── admin
│   │   │   └── index.html
│   │   ├── dashboard
│   │   │   └── dashboard.html
│   │   ├── frontend
│   │   │   ├── change_password.html
│   │   │   ├── contact_us.html
│   │   │   ├── landing.html
│   │   │   ├── login.html
│   │   │   ├── reset_password.html
│   │   │   └── signup.html
│   │   ├── layouts
│   │   │   ├── banner.html
│   │   │   ├── base.html
│   │   │   ├── footer.html
│   │   │   └── header.html
│   │   ├── macros
│   │   │   ├── _confirm_account.html
│   │   │   ├── _flash_msg.html
│   │   │   ├── _form.html
│   │   │   └── _reset_password.html
│   │   ├── settings
│   │   │   ├── password.html
│   │   │   └── profile.html
│   │   └── tasks
│   │       ├── add_task.html
│   │       ├── contribute.html
│   │       ├── edit_task.html
│   │       ├── landing.html
│   │       ├── my_tasks.html
│   │       ├── naso_tableview.html
│   │       ├── show_plot.html
│   │       ├── show_scfeature.html
│   │       ├── table_view.html
│   │       └── view_task.html
│   ├── user
│   │   ├── constants.py
│   │   ├── __init__.py
│   │   └── models.py
│   └── utils.py
├── manage.py
├── README.md
├── requirements.txt
├── run.old.sh
├── run.sh
├── screenshots
│   ├── admin.png
│   ├── dashboard.png
│   ├── homepage.png
│   ├── login.png
│   ├── profile.png
│   ├── signup.png
│   └── tasks.png
└── tests
    ├── __init__.py
    └── test_flaskstarter.py
```


## Modules

This application uses the following modules

- Flask[async]
- Flask-SQLAlchemy
- Flask-WTF
- Flask-Mail
- Flask-Caching
- Flask-Login
- Flask-Admin
- email-validator
- itsdangerous
- WTForms==2.3.3
- pytest
- pandas
- plotly
- numpy
- pymongo
- shortuuid
- seaborn
- dashbio
- gunicorn


## Testing

Note: This web application has been tested thoroughly during multiple large projects, however tests for this bare minimum version would be added in `tests` folder very soon to help you get started.


## Features

- Flask 2.0, Python (`PEP8`)
- Signup, Login with (email, password)
- Forget/reset passwords
- Email verification
- User profile/password updates
- User roles (admin, user, staff)
- User profile status (active, inactive)
- Admin dashboard for management
- Contact us form
- Bootstrap template (minimal)
- Utility scripts (initiate dummy database, run test server)
- Test & Production Configs


## Flask 2.0 `async` or not `async`

 - asynchronous support in Flask 2.0 is an amazing feature
 - however, use it only when it has a clear advantage over the equivalent synchronous code
 - write asynchronous code, if your application's routes, etc. are making heavy I/O-bound operations, like:
    - sending emails, making API calls to external servers, working with the file system, etc
 - otherwise, if your application is doing CPU-bound operations or long-running tasks, like:
    - processing images or large files, creating backups or running AI/ML models, etc
    - it is advised to use tools like "Celery" or "Huey", etc.


## `async` demo in our application

Check `emails/__init__.py` to see how emails being sent in `async` mode


## Need Help? 🤝

If you need further help, reach out to me via [Twitter](https://twitter.com/kundan7_) DM.
