ps ux | grep '0.0.0.0:8000' | grep -v grep | awk '{print $2}' | xargs kill -9
