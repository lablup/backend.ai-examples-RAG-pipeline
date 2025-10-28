#!/bin/bash

set -e

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
elif [ -f ".env.template" ]; then
    set -a
    cp .env.template .env
    source .env
    set +a
fi

cd $TASK_DIR
pip install -r requirements.txt
python3 main.py