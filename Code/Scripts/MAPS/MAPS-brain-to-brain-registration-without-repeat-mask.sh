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
#  Last Changed      : $LastChangedDate: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $ 
#  Revision          : $Revision: 3326 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

function brain_to_brain_registration_without_repeat_mask_using_air()
{
  local baseline_image=$1
  local baseline_region=$2
  local repeat_image=$3
  local output_reg_brain_image=$4
  local output_reg_brain_series_number=$5
  local output_reg_air=$6
  local is_invert_air=$7

  local tmp_dir=`mktemp -d -q ~/temp/hippo-template-match-reg.XXXXXX`
  
  local whole_brain_initfile=${tmp_dir}/whole_brain_reg.ini
  makemask ${baseline_image} ${baseline_region} ${tmp_dir}/baseline_brain_mask -d 2
  t1=`imginfo ${baseline_image} -tanz 0.2`
  t2=`imginfo ${repeat_image} -tanz 0.2`
  ${AIR_BIN}/alignlinear ${baseline_image} ${repeat_image} ${output_reg_air} -g ${whole_brain_initfile} y -m 12 \
     -e1 ${tmp_dir}/baseline_brain_mask -p1 1 -p2 1 -s 81 1 3 -c 0.0001 -h 200 -r 200 -q -x 1 \
     -t1 $t1 -t2 $t2 -v -b1 2 2 2 -b2 2 2 2
  rm -f ${output_reg_air}
  ${AIR_BIN}/alignlinear ${baseline_image} ${repeat_image} ${output_reg_air} -m 12 \
     -e1 ${tmp_dir}/baseline_brain_mask \
     -f ${whole_brain_initfile} -p1 1 -p2 1 -s 2 1 2 -c 0.0001 -h 200 -r 200 -q -x 1 -t1 $t1 -t2 $t2 -v 
     
  if [ "${is_invert_air}" == "no" ]
  then 
    ${AIR_BIN}/reslice ${output_reg_air} ${output_reg_brain_image} -k -n 10 -o
  else
    ${AIR_BIN}/invert_air ${output_reg_air} ${output_reg_air} y
    ${AIR_BIN}/reslice ${output_reg_air} ${output_reg_brain_image} -k -n 10 -o
  fi 
  rm -rf ${tmp_dir}
  
  return 0
}

function brain_to_brain_registration_without_repeat_mask_using_irtk()
{
  local baseline_image=$1
  local baseline_region=$2
  local repeat_image=$3
  local output_reg_brain_image=$4
  local output_reg_brain_series_number=$5
  local output_reg_air=${6}
  local is_invert_air=$7

  local temp_dir=`mktemp -d -q ~/temp/__areg.XXXXXXXX`
  
  if [ ! -f ${output_reg_air} ] 
  then 
  
    # Registration. 
    local parameter_file=`mktemp ~/temp/param.XXXXXXXXXX`
    echo "Target blurring (in mm) = 0 "  > ${parameter_file}
    echo "Target resolution (in mm) = 0"  >> ${parameter_file}
    echo "# source image paramters"  >> ${parameter_file}
    echo "Source blurring (in mm) = 0"  >> ${parameter_file}
    echo "Source resolution (in mm)  = 0"  >> ${parameter_file}
    echo "# registration parameters"  >> ${parameter_file}
    echo "No. of resolution levels = 3"  >> ${parameter_file}
    echo "No. of bins = 128"  >> ${parameter_file}
    echo "No. of iterations = 100"  >> ${parameter_file}
    echo "No. of steps = 2"  >> ${parameter_file}
    echo "Length of steps = 1"  >> ${parameter_file}
    echo "Similarity measure = CC"  >> ${parameter_file}
    ${MIDAS_FFD}/ffdareg.sh ${baseline_image} ${repeat_image} ${output_reg_air} -dof 12 -comreg -params ${parameter_file} -tmpdir ${temp_dir}
    rm -f ${parameter_file}
    
    local parameter_file=`mktemp ~/temp/param.XXXXXXXXXX`
    echo "Target blurring (in mm) = 0 "  > ${parameter_file}
    echo "Target resolution (in mm) = 0"  >> ${parameter_file}
    echo "# source image paramters"  >> ${parameter_file}
    echo "Source blurring (in mm) = 0"  >> ${parameter_file}
    echo "Source resolution (in mm)  = 0"  >> ${parameter_file}
    echo "# registration parameters"  >> ${parameter_file}
    echo "No. of resolution levels = 1"  >> ${parameter_file}
    echo "No. of bins = 128"  >> ${parameter_file}
    echo "No. of iterations = 100"  >> ${parameter_file}
    echo "No. of steps = 4"  >> ${parameter_file}
    echo "Length of steps = 2"  >> ${parameter_file}
    echo "Similarity measure = CC"  >> ${parameter_file}
    ${MIDAS_FFD}/ffdareg.sh ${baseline_image} ${repeat_image} ${output_reg_air} -troi ${baseline_region} -dof 12 -inidof ${output_reg_air} -params ${parameter_file} -tmpdir ${temp_dir}
    rm -f ${parameter_file}
  fi 
  
  # Transform the image. 
  if [ "${is_invert_air}" == "yes" ]
  then 
    local invert_flag="-invert"
    #${MIDAS_FFD}/ffdtransformation.sh  ${baseline_image} ${repeat_image} ${output_reg_brain_image} ${output_reg_air} -cspline -invert -tmpdir ${temp_dir}
    ${MIDAS_FFD}/ffdtransformation.sh  ${baseline_image} ${repeat_image} ${output_reg_brain_image} ${output_reg_air} -sinc -invert -tmpdir ${temp_dir}
    local orientation=`imginfo ${repeat_image} -orient`
    anchange ${output_reg_brain_image} ${output_reg_brain_image} -setorient ${orientation}
  else
    ${MIDAS_FFD}/ffdtransformation.sh ${repeat_image} ${baseline_image} ${output_reg_brain_image} ${output_reg_air} -cspline -tmpdir ${temp_dir}
    local orientation=`imginfo ${baseline_image} -orient`
    anchange ${output_reg_brain_image} ${output_reg_brain_image} -setorient ${orientation}
  fi 
  
  rm ${temp_dir} -rf 
  
  return 0
}



