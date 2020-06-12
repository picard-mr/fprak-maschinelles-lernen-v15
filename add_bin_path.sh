#!/bin/bash

# copied from
# https://stackoverflow.com/a/50305652
ipython profile create
cat >/home/jovyan/.ipython/profile_default/ipython_config.py <<EOF
c.InteractiveShellApp.exec_lines = [
    'import sys; sys.path.append("/home/netter/bin")'
]
EOF
