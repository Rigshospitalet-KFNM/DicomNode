#/bin/bash

current_directory=$(pwd)

if [ ! -d report_data ]; then
  mkdir -p report_data
fi

if [ ! -e report_data/report_image.png ]; then
  wget -P report_data -O report_data/report_image.png https://www.rigshospitalet.dk/SiteCollectionImages/Logo_Rigshospitalet_RGB_54px.png
fi

fc-list | grep -q "Mari"
status=$?

if [ $status -eq 0 ]; then
  echo "Using Mari font"
  export DICOMNODE_ENV_FONT="Mari"
fi

export DICOMNODE_ENV_REPORT_DATA_PATH="$current_directory/report_data/"

coverage run runtests.py $@
coverage report --show-missing --omit=/lib/tests/*,tests/*,runtests.py,venv/* --skip-covered

if [ -d coverage ]; then
  mkdir -p coverage
fi
coverage-lcov --output_file_path coverage/lcov.info
