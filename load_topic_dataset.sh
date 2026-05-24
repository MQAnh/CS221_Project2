#!/bin/bash

cd data
echo "Downloading data.zip..."

wget -O data.zip \
https://github.com/kh4nh12/UIT-ViON-Dataset/raw/main/data.zip

echo "Unzipping..."

unzip data.zip

echo "Done."

cd ..