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
#  Last Changed      : $LastChangedDate: 2010-06-11 11:25:48 +0100 (Fri, 11 Jun 2010) $ 
#  Revision          : $Revision: 3373 $
#  Last modified by  : $Author: kkl $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# Debug
#set -x


#
# Script to do groupwise registration. 
# 

# Default values used on img-135 by Kelvin. 
if [ ! -n "${MIDAS_BIN}" ]
then 
  export MIDAS_BIN=/var/NOT_BACKUP/work/midas/build/bin
fi 
if [ ! -n "${FFITK_BIN}" ]
then
  export FFITK_BIN=/var/NOT_BACKUP/work/ffitk/bin
fi 
if [ ! -n "${MIDAS_FFD}" ]
then
  export MIDAS_FFD=/var/NOT_BACKUP/work/midasffd
fi   
if [ ! -n "${FSLDIR}" ]
then 
  export FSLDIR=/var/lib/midas/pkg/i686-pc-linux-gnu/fsl
fi   
if [ ! -n "${F3D_BIN}" ]
then 
  export F3D_BIN=/home/samba/user/leung/work/nifti-reg/nifty_reg-1.2/reg-apps
fi   

# No undefined varaiable. 
set -u 


# Set up all the essential programs. 
export nifti_reg=${F3D_BIN}/reg_f3d
export nifti_resample=${F3D_BIN}/reg_resample
export nifti_tools=${F3D_BIN}/reg_tools 
export abs_filter=niftkAbsImageFilter
export convert=niftkConvertImage

# Input file containing the images and the regions to be groupwisely registered. 
input_file=$1
dilation=$2
cps=$3
output_dir=$4

initial_atlas=""
initial_mask=""
atlas=""
identity_dof=${output_dir}/identity_dof.nii

#
# Pick the initial atlas. 
#
each_line=`head -n 1 ${input_file}`
input_image=`echo ${each_line} | awk '{printf $1}'`
input_region=`echo ${each_line} | awk '{printf $2}'`
input_image_basename=`basename ${input_image}`
input_study_id=`echo ${input_image_basename} | awk -F- '{printf $1}'`
input_image_dilated=${output_dir}/${input_study_id}-dilated.img
initial_mask=${output_dir}/${input_study_id}-initial_mask.img

initial_atlas=${input_image_dilated}

makemask ${input_image} ${input_region} ${initial_mask} -d ${dilation}

# Create an identity f3d dof. 
if [ ! -f "${identity_dof}" ]
then 
  makemask ${input_image} ${input_region} ${input_image_dilated} -k -bpp 16 -d ${dilation}
  ${nifti_reg} -target ${input_image} -source ${input_image} -cpp ${identity_dof} -ln 3 -sx ${cps} 
fi
echo ${atlas}

atlas=${output_dir}/atlas.img
if [ ! -f "${atlas}" ] 
then 
  cp ${input_image_dilated} ${atlas}
  cp ${input_image_dilated%.img}.hdr ${atlas%.img}.hdr
fi 

#
# All images register to the atlas. 
#
count=0
cat ${input_file} | while read each_line
do
  input_image=`echo ${each_line} | awk '{printf $1}'`
  input_region=`echo ${each_line} | awk '{printf $2}'`
  input_image_basename=`basename ${input_image}`
  input_study_id=`echo ${input_image_basename} | awk -F- '{printf $1}'`
  input_image_dilated=${output_dir}/${input_study_id}-dilated.img
  
  (( count++ ))
  echo "count=${count}"
  
  makemask ${input_image} ${input_region} ${input_image_dilated} -k -bpp 16 -d ${dilation}
  
  # Register to the current atlas. 
  previous_dof=${output_dir}/${input_study_id}-previous-dof.nii
  input_cpp=""
  output_dof=${output_dir}/${input_study_id}-current-dof.nii
  output_image=${output_dir}/${input_study_id}-current-output.nii
  if [ -f "${previous_dof}" ]
  then 
    input_cpp="-incpp ${previous_dof} -ln 1"
  else
    input_cpp="-ln 3"
  fi 
  ${nifti_reg} -target ${atlas} -source ${input_image_dilated} -cpp ${output_dof} -result ${output_image} -sx ${cps} ${input_cpp} 
done

#
# Calculate the average deformation. 
#
count=0
average_deformation=${output_dir}/average_dof.nii
cat ${input_file} | while read each_line
do
  input_image=`echo ${each_line} | awk '{printf $1}'`
  input_region=`echo ${each_line} | awk '{printf $2}'`
  input_image_basename=`basename ${input_image}`
  input_study_id=`echo ${input_image_basename} | awk -F- '{printf $1}'`
  input_image_dilated=${output_dir}/${input_study_id}-dilated.img
  deformation_dof=${output_dir}/${input_study_id}-deformation.nii
  
  (( count++ ))
  ${nifti_tools} -in ${output_dir}/${input_study_id}-current-dof.nii -out ${deformation_dof} -sub ${identity_dof}
  
  if [ ${count} == 1 ] 
  then 
    cp ${deformation_dof} ${average_deformation}
  else
    ${nifti_tools} -in ${average_deformation} -out ${average_deformation} -add ${deformation_dof}
  fi 
done
count=`wc ${input_file} | awk '{printf $1}'`
${nifti_tools} -in ${average_deformation} -out ${average_deformation} -divV ${count}


#
# Update all the current transformations. 
#
count=0
cat ${input_file} | while read each_line
do
  input_image=`echo ${each_line} | awk '{printf $1}'`
  input_region=`echo ${each_line} | awk '{printf $2}'`
  input_image_basename=`basename ${input_image}`
  input_study_id=`echo ${input_image_basename} | awk -F- '{printf $1}'`
  previous_dof=${output_dir}/${input_study_id}-previous-dof.nii
  output_dof=${output_dir}/${input_study_id}-current-dof.nii
  transformed_image=${output_dir}/${input_study_id}-transformed.img
  input_image_dilated=${output_dir}/${input_study_id}-dilated.img
  
  (( count++ ))
  
  ${nifti_tools} -in ${output_dof} -out ${previous_dof} -sub ${average_deformation} 
  
  ${nifti_resample} -target ${initial_atlas} -source ${input_image_dilated} -cpp ${previous_dof} -result ${transformed_image} -TRI
  
  niftkKMeansWindowWithLinearRegressionNormalisationBSI \
      ${initial_atlas} ${initial_mask} \
      ${transformed_image} ${transformed_image} \
      ${initial_atlas} ${initial_mask} \
      ${transformed_image} ${transformed_image} \
      1 1 1 -1 -1 temp.hdr temp.hdr ${transformed_image} 
  
  if [ ${count} != 1 ] 
  then 
    factor=`echo "(${count}-1)/${count}" | bc -l`
    ${nifti_tools} -in ${atlas} -out ${atlas} -mulV ${factor}
    ${nifti_tools} -in ${transformed_image} -out ${transformed_image} -divV ${count}
    ${nifti_tools} -in ${atlas} -out ${atlas} -add ${transformed_image} 
  else
    ${nifti_tools} -in ${transformed_image} -out ${atlas} -divV ${count}
  fi 
  
done



























