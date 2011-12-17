#!/bin/bash

#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2010-08-24 16:48:21 +0100 (Tue, 24 Aug 2010) $ 
#  Revision          : $Revision: 3748 $
#  Last modified by  : $Author: kkl $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/
#
# Script to run a groupwise DBC. 
# 

source _niftkCommon.sh

set -ux

function Usage()
{
cat <<EOF

Script to compute symmetric transform and BSI.

Usage: $0 target.img target.roi target.dof source.img source.roi source.dof output_dir [options]

Mandatory Arguments:

  target.img        : target (or baseline) image
  target.roi        : target (or baseline) region
  target.dof        : target (or baseline) dof
  source.img        : source (or repeat) image
  source.roi        : source (or repeat) region
  source.dof        : source (or repeat) dof

  output_dir        : Output directory
                       
EOF
exit 127
}

asym_dof=""
interpolation=4
double_window="yes"
ss_atlas=""

# Parse command line options
while [ "$#" -gt 0 ]
do
    case $1 in
    -asym)
      asym_dof="${identity_dof} 1 ${identity_dof} 1"  
      ;;
    -interpolation)
      interpolation=$2
      shift 1
      ;;
    -double_window)
      double_window=$2
      shift 1
      ;;
    -ss_atlas)
      ss_atlas=$2
      shift 1
      ;; 
    *)
      echo "break"
      break
      ;;
    esac
    shift 1
done

# Get parameters
output_dir=${1}
output_prefix=${2}
number_of_images=${3}
shift 3

mkdir -p $output_dir
execute_command "tmpdir=`mktemp -d -q ~/temp/compute_symmetric_transform.XXXXXX`"
execute_command "mkdir -p ${tmpdir}/${output_dir}"
function cleanup
{
  echo "Cleaning up..."
  execute_command "rm -rf  ${tmpdir}"
}
trap "cleanup" EXIT SIGINT SIGTERM SIGKILL 

identity_dof=${tmpdir}/identity.dof

for ((i=1; i<=${number_of_images}; i++))
do 
  image[${i}]=$1
  shift 1
  region[${i}]=$1
  shift 1
  local_region[${i}]=$1
  shift 1
  air_file[${i}]=$1
  shift 1
  
  for ((j=1; j<=${number_of_images}; j++))
  do
    if [ ${i} != ${j} ] 
    then 
      index=`echo "${i}*${number_of_images}+${j}" | bc`
      dof[${index}]=$1
      dof_star[${index}]=${tmpdir}/dof_${i}_${j}_star.dof
      shift 1
    fi 
  done
  
done  



niftkCreateTransformation -type 3 -ot ${identity_dof}  0 0 0 0 0 0 1 1 1 0 0 0

function compute_star()
{
  local _dof_1_2=$1
  local _dof_2_1=$2
  local _dof_1_2_star=$3
  local _dof_1_2_star_inverse=$4

  # Get the inverse of repeat transform.
  local _dof_2_1_inverse=${tmpdir}/__dof_2_1_inverse
  itkInvertTransformation ${_dof_2_1} ${_dof_2_1_inverse}

  # Compute the mean transform for baseline image.
  itkComputeMeanTransformation ${_dof_1_2_star} 1e-8 ${_dof_1_2} 1 ${_dof_2_1_inverse} 1 ${asym_dof}
  itkInvertTransformation ${_dof_1_2_star} ${_dof_1_2_star_inverse}
}

function transform()
{
  local _image1=$1
  local _region1=$2
  local _image2=$3
  local _image3=$4
  local _dof_1_2=$5
  local _dof_1_3=$6
  local _output_prefix=$7
  
  # Compute the mean transform for baseline image.
  local average_transform=${output_dir}/${_output_prefix}_average.dof
  itkComputeMeanTransformation ${average_transform} 1e-8 ${_dof_1_2} 2 ${_dof_1_3} 2 ${identity_dof} 1
  
  # Do the transform.
  local resliced_image=${output_dir}/${_output_prefix}.img
  local resliced_mask=${output_dir}/${_output_prefix}_mask
  local region1_img=${tmpdir}/baseline_mask.img
  local resliced_region1_img=${output_dir}/${_output_prefix}_mask.img
  
  makemask ${_image1} ${_region1} ${region1_img}
  niftkTransformation -ti ${_image1} -o ${resliced_image} -j ${interpolation} -g ${average_transform} -sym_midway 3 ${_image1} ${_image2} ${_image3} -invertAffine
  niftkAbsImageFilter -i ${resliced_image} -o ${resliced_image}
  niftkTransformation -ti ${region1_img} -o ${resliced_region1_img} -j 2 -g ${average_transform} -sym_midway 3 ${_image1} ${_image2} ${_image3} -invertAffine
  makeroi -img ${resliced_region1_img} -out ${resliced_mask} -alt 128
}

if [ 1 == 1 ]
then 

# Create the pairwise symmetric dof files. 
for ((i=1; i<=${number_of_images}; i++))
do 
  (( j=i+1 ))
  for ((; j<=${number_of_images}; j++))
  do 
    index1=`echo "${i}*${number_of_images}+${j}" | bc`
    index2=`echo "${j}*${number_of_images}+${i}" | bc`
    compute_star ${dof[${index1}]} ${dof[${index2}]} ${dof_star[${index1}]} ${dof_star[${index2}]}
  done
done  

# Transform the images to a middle space. 
for ((i=1; i<=${number_of_images}; i++))
do 
  image_arguments="${image[${i}]}"
  dof_arguments=""
  for ((j=1; j<=${number_of_images}; j++))
  do 
    if [ ${i} != ${j} ]
    then 
      index=`echo "${i}*${number_of_images}+${j}" | bc`
      image_arguments="${image_arguments} ${image[${j}]}"
    fi 
  done
  for ((j=1; j<=${number_of_images}; j++))
  do 
    if [ ${i} != ${j} ]
    then 
      index=`echo "${i}*${number_of_images}+${j}" | bc`
      dof_arguments="${dof_arguments} ${dof_star[${index}]} 2"
    fi 
  done
    
  #transform ${arguments} ${output_prefix}_${i}

  # Compute the mean transform for baseline image.
  average_transform=${output_dir}/${output_prefix}_${i}_average.dof
  itkComputeMeanTransformation ${average_transform} 1e-8 ${dof_arguments} ${identity_dof} 1

  # Do the transform.
  resliced_image=${output_dir}/${output_prefix}_${i}.img
  resliced_mask=${output_dir}/${output_prefix}_${i}_mask
  region1_img=${tmpdir}/baseline_mask.img
  resliced_region1_img=${output_dir}/${output_prefix}_${i}_mask.img

  niftkTransformation -ti ${image[${i}]} -o ${resliced_image} -j ${interpolation} -g ${average_transform} -sym_midway ${number_of_images} ${image_arguments} -invertAffine
  niftkAbsImageFilter -i ${resliced_image} -o ${resliced_image}
  makemask ${image[${i}]}  ${region[${i}]} ${region1_img}
  niftkTransformation -ti ${region1_img} -o ${resliced_region1_img} -j 2 -g ${average_transform} -sym_midway  ${number_of_images} ${image_arguments} -invertAffine
  makeroi -img ${resliced_region1_img} -out ${resliced_mask} -alt 128
  
  if [ -f "${local_region[${i}]}" ] && [ -f "${air_file[${i}]}" ] 
  then 
    if [ "${local_region[${i}]%.img}" != "${local_region[${i}]}" ] 
    then 
      local_region_img=${local_region[${i}]}
    else
      local_region_img=${tmpdir}/${output_prefix}_${i}_local.img
      makemask ${ss_atlas} ${local_region[${i}]} ${local_region_img}
    fi 
    resliced_local_region1_img=${output_dir}/${output_prefix}_${i}_mask_local.img
    resliced_local_region1_dof=${output_dir}/${output_prefix}_${i}_mask_local.dof
    resliced_local_mask=${output_dir}/${output_prefix}_${i}_local_mask
    niftkMultiplyTransformation -i2 ${air_file[${i}]} -i1 ${average_transform} -o1 ${resliced_local_region1_dof}
    niftkTransformation -ti ${local_region_img} -o ${resliced_local_region1_img} -j 2 -g ${resliced_local_region1_dof} -sym_midway  ${number_of_images} ${image_arguments} -invertAffine
    makeroi -img ${resliced_local_region1_img} -out ${resliced_local_mask} -alt 128
  fi 
done  


fi 

# Differential bias correction. 
arguments=""
for ((i=1; i<=${number_of_images}; i++))
do 
  arguments="${arguments} ${output_dir}/${output_prefix}_${i}.hdr ${output_dir}/${output_prefix}_${i}_mask.img ${output_dir}/${output_prefix}_${i}_dbc.hdr"
done  
niftkMTPDbc -mode 1 ${arguments} 

# Compute BSI. 
for ((i=1; i<=${number_of_images}; i++))
do 
  (( j=i+1 ))
  for ((; j<=${number_of_images}; j++))
  do 
    if [ ! -f "${local_region[${i}]}" ] || [ ! -f "${air_file[${i}]}" ] 
    then 
      compute-kmeans-bsi.sh ${output_dir}/${output_prefix}_${i}_dbc ${output_dir}/${output_prefix}_${i}_mask \
                        ${output_dir}/${output_prefix}_${j}_dbc ${output_dir}/${output_prefix}_${j}_mask \
                        ${output_dir}
    else
      makemask ${output_dir}/${output_prefix}_${i}_dbc.img ${output_dir}/${output_prefix}_${i}_mask ${output_dir}/${output_prefix}_${i}_mask.img
      makemask ${output_dir}/${output_prefix}_${j}_dbc.img ${output_dir}/${output_prefix}_${j}_mask ${output_dir}/${output_prefix}_${j}_mask.img
      makemask ${output_dir}/${output_prefix}_${i}_dbc.img ${output_dir}/${output_prefix}_${i}_local_mask ${output_dir}/${output_prefix}_${i}_local_mask.img
      makemask ${output_dir}/${output_prefix}_${j}_dbc.img ${output_dir}/${output_prefix}_${j}_local_mask ${output_dir}/${output_prefix}_${j}_local_mask.img
      
      if  [ "${double_window}" == "yes" ] 
      then 
        niftkKNDoubleWindowBSI \
          ${output_dir}/${output_prefix}_${i}_dbc.img ${output_dir}/${output_prefix}_${i}_mask.img \
          ${output_dir}/${output_prefix}_${j}_dbc.img ${output_dir}/${output_prefix}_${j}_mask.img \
          ${output_dir}/${output_prefix}_${i}_dbc.img ${output_dir}/${output_prefix}_${i}_local_mask.img \
          ${output_dir}/${output_prefix}_${j}_dbc.img ${output_dir}/${output_prefix}_${j}_local_mask.img \
          1 1 3   \
          ${tmpdir}/bs.img ${tmpdir}/rs.img \
          dummy ${tmpdir}/xor.img ${output_dir}/${output_prefix}_${i}_local_mask.img ${output_dir}/${output_prefix}_${j}_local_mask.img \
          1.0 0.5 dummy 0.05 > ${output_dir}/${output_prefix}_${i}_dbc-${output_prefix}_${j}_dbc_local.qnt
      else
          niftkKMeansWindowWithLinearRegressionNormalisationBSI \
            ${output_dir}/${output_prefix}_${i}_dbc.img ${output_dir}/${output_prefix}_${i}_mask.img \
            ${output_dir}/${output_prefix}_${j}_dbc.img ${output_dir}/${output_prefix}_${j}_mask.img \
            ${output_dir}/${output_prefix}_${i}_dbc.img ${output_dir}/${output_prefix}_${i}_local_mask.img \
            ${output_dir}/${output_prefix}_${j}_dbc.img ${output_dir}/${output_prefix}_${j}_local_mask.img \
            1 1 3 -1 -1 \
            ${tmpdir}/bs.img ${tmpdir}/rs.img ${tmpdir}/norm.img > ${output_dir}/${output_prefix}_${i}_dbc-${output_prefix}_${j}_dbc_local.qnt
      fi 
        
      rm -f ${output_dir}/${output_prefix}_${i}_local_mask.* ${output_dir}/${output_prefix}_${j}_local_mask.* 
    fi 
  done                      
done

# Clean up junks. 
for ((i=1; i<=${number_of_images}; i++))
do 
  rm -f ${output_dir}/${output_prefix}_${i}_dbc.* 
  rm -f ${output_dir}/${output_prefix}_${i}_mask.*
  rm -f ${output_dir}/${output_prefix}_${i}_mask_local.* 
done   

     
execute_command "rm ${tmpdir} -rf"










