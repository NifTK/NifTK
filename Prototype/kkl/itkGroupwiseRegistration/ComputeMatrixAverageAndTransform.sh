#!/bin/bash

set -x

temp_dir_prefix=~/temp
if [ "${MY_TEMP}" != "" ]
then 
  temp_dir_prefix=${MY_TEMP}
fi 

tmp_dir=`mktemp -d -q ${temp_dir_prefix}/_compute_average.XXXXXX`
function cleanup
{
  echo "Cleaning up..."
  rm -rf  ${tmp_dir}
}
trap "cleanup" EXIT SIGINT SIGTERM SIGKILL 

moving_image=$1
output_image=$2
output_transform=$3
nubmer_of_images=$4
is_region=$5
starting_arg=6

for (( i=0; i<$nubmer_of_images; i++ ))
do
  arg=$(( starting_arg+i ))
  input_images[${i}]=${!arg}
done   

starting_arg=$(( starting_arg+nubmer_of_images ))

all_transforms=""
for (( i=0; i<$nubmer_of_images; i++ ))
do 
  arg=$(( starting_arg+i*2 ))
  input_transform[${i}]=${!arg}
  (( arg++ ))
  if [ "${!arg}" == "1" ] 
  then 
    transform=${tmp_dir}/transform${i}.dof 
    itkInvertTransformation ${input_transform[${i}]} ${transform}
  else
    transform=${input_transform[${i}]}
  fi 
  all_transforms="${all_transforms} ${transform}"
done

itkComputeMeanTransformation ${output_transform} 1e-9 ${all_transforms}

if [ "${is_region}" == "no" ]
then 
  niftkTransformation -ti ${moving_image} -o ${output_image} -j 2 -g ${output_transform} -sym_midway ${input_images[0]} ${input_images[1]} 
  niftkAbsImageFilter -i ${output_image} -o ${output_image}
else
  mask_img=${tmp_dir}/mask.hdr
  niftkTransformation -ti ${moving_image} -o ${mask_img} -j 2 -g ${output_transform} -sym_midway ${input_images[0]} ${input_images[1]} 
  makeroi -img ${mask_img} -out ${output_image} -alt 128
fi 


