#/bin/bash

coverage run runtests.py
coverage report --show-missing --omit=/lib/tests/*,tests/*,runtests.py,venv/* --skip-covered
[ -d coverage ] || mkdir -p coverage
coverage-lcov --output_file_path coverage/lcov.info
