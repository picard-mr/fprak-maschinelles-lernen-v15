#!/bin/bash

cat >/home/jovyan/.ipython/profile_default/ipython_config.py <<EOF
c.InteractiveShellApp.exec_lines = [
    'import sys; sys.path.append("/home/netter/bin")'
]
EOF
