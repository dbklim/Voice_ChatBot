#!/bin/sh

export $(dbus-launch)
python3 rest_server.py 0.0.0.0:5000
