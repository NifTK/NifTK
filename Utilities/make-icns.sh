#!/bin/bash

function print_usage() {
  echo "

Use to convert a 1024 x 1024 sized .png file into a series of icons stored in a .icns file.

Usage:
  make-icns.sh inputImage1024x1024.png 
"
  exit 1
}

if [ $# -ne 1 ]
then
  print_usage
fi

input_file=$1

if [ ! -e $input_file ]; then
  echo "ERROR: Input file does not exist"
fi

tmp_dir=/tmp/MyIcon.iconset

if [ -d $tmp_dir ]; then
  rm -rf $tmp_dir
fi
 
mkdir $tmp_dir

sips -z 16 16     $input_file --out $tmp_dir/icon_16x16.png
sips -z 32 32     $input_file --out $tmp_dir/icon_16x16@2x.png
sips -z 32 32     $input_file --out $tmp_dir/icon_32x32.png
sips -z 64 64     $input_file --out $tmp_dir/icon_32x32@2x.png
sips -z 128 128   $input_file --out $tmp_dir/icon_128x128.png
sips -z 256 256   $input_file --out $tmp_dir/icon_128x128@2x.png
sips -z 256 256   $input_file --out $tmp_dir/icon_256x256.png
sips -z 512 512   $input_file --out $tmp_dir/icon_256x256@2x.png
sips -z 512 512   $input_file --out $tmp_dir/icon_512x512.png

cp $input_file $tmp_dir/icon_512x512@2x.png

iconutil -c icns $tmp_dir --output icon.icns

rm -rf $tmp_dir


