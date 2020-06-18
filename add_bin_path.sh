#!/bin/bash

# csbdeep, stardist and n2v are already installed
# I want to use my own repos of those, so deinstall them
pip uninstall n2v
pip uninstall stardist
pip uninstall csbdeep

# copied from
# https://stackoverflow.com/a/50305652
ipython profile create
cat >/home/jovyan/.ipython/profile_default/ipython_config.py <<EOF
c.InteractiveShellApp.exec_lines = [
    'import sys; sys.path.append("/home/netter/bin")'
]
EOF
