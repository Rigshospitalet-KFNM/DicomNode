#!/bin/bash

source ../venv/bin/activate
python -m sphinx -T -E -b html -d build/doctrees -D language=en source build/html
