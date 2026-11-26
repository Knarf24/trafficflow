#!/bin/sh
# Puts the web page, the Python package, the data and the detector model together in _site/
# (what GitHub Pages serves).
set -e
rm -rf _site
mkdir -p _site/trafficflow _site/models
cp -r web/. _site/
cp trafficflow/*.py _site/trafficflow/
cp data/synthetic_drive_through.csv _site/data/
cp models/object_detection_nanodet_2022nov.onnx _site/models/
touch _site/.nojekyll
