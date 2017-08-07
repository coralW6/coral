ps ux | grep '0.0.0.0:8000' | grep -v grep | awk '{print $2}' | xargs kill -9
nohup python2.7 manage.py runserver 0.0.0.0:8000 >> logs/coral.log 2>&1 &
