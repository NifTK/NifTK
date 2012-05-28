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
#  Last Changed      : $LastChangedDate: 2011-08-18 13:56:15 +0100 (Thu, 18 Aug 2011) $ 
#  Revision          : $Revision: 7129 $
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
# Common functions used MAPS. 
# 
echo "Helper functions for hippo segmentation" 

#
# Simple brain to brain registration with 12-dof. 
#
function brain_to_brain_registration()
{
  local baseline_image=$1
  local baseline_region=$2
  local repeat_image=$3
  local repeat_region=$4
  local output_reg_brain_image=$5
  local output_reg_brain_region=$6
  local output_reg_brain_series_number=$7
  local output_reg_air=$8

  local tmp_dir=`mktemp -d -q /usr/tmp/hippo-template-match-reg.XXXXXX`
  
  prealign=${tmp_dir}/prealign.ini
  local whole_brain_initfile=${tmp_dir}/whole_brain_reg.ini
  makemask ${baseline_image} ${baseline_region} ${tmp_dir}/baseline_brain_mask
  makemask ${repeat_image} ${repeat_region} ${tmp_dir}/repeat_brain_mask -d 2
  reg_prealign ${baseline_image} ${repeat_image} ${baseline_region} ${repeat_region} $prealign -12 -t1 0.2 -t2 0.2 -a -v
  t1=`imginfo ${baseline_image} -tanz 0.2 -roi ${baseline_region}`
  t2=`imginfo ${repeat_image} -tanz 0.2 -roi ${repeat_region}`
  ${AIR_BIN}/alignlinear ${baseline_image} ${repeat_image} ${output_reg_air} -f ${prealign} -g ${whole_brain_initfile} y -m 12 \
     -e1 ${tmp_dir}/baseline_brain_mask -e2 ${tmp_dir}/repeat_brain_mask -p1 1 -p2 1 -s 81 1 3 -c 0.000001 -h 200 -r 200 -q -x 1 \
     -t1 $t1 -t2 $t2 -v -b1 2 2 2 -b2 2 2 2
  rm -f ${output_reg_air}
  ${AIR_BIN}/alignlinear ${baseline_image} ${repeat_image} ${output_reg_air} -m 12 \
     -e1 ${tmp_dir}/baseline_brain_mask -e2 ${tmp_dir}/repeat_brain_mask  \
     -f ${whole_brain_initfile} -p1 1 -p2 1 -s 2 1 2 -c 0.0000001 -h 200 -r 200 -q -x 1 -t1 $t1 -t2 $t2 -v 
     
  ${AIR_BIN}/reslice ${output_reg_air} ${output_reg_brain_image} -k -n 10 -o
  regslice ${output_reg_air} ${subject_brain_region} ${output_reg_brain_region} ${output_reg_brain_series_number}
  rm -rf ${tmp_dir}
  
  return 0
}

#
# Delineate one hippo using FFD. 
#
function hippo_delineation()
{
  echo "number of arguments=$#"
  
  local subject_image=$1 
  local subject_brain_region=$2
  local template_image=$3
  local template_brain_region=$4
  local output_areg_template_brain_image=$5
  local output_areg_template_brain_region=$6
  local output_areg_template_brain_series_number=$7
  local output_areg_template_air=$8
  local template_hippo_region=$9
  local output_areg_hippo_region=${10}
  local output_delineate_dir=${11}
  local output_study_id=${12}
  local output_series_number=${13}
  local output_echo_number=${14}
  local output_local_areg_hippo_region=${15}
  local output_nreg_hippo_region=${16}
  local output_nreg_template_hippo_image=${17}
  local output_nreg_template_hippo_dof_16=${18}
  local output_nreg_template_hippo_dof_8=${19}
  local output_nreg_template_hippo_dof_4=${20}
  local output_areg_mm_hippo_region=${21}
  local output_nreg_mm_hippo_region=${22}
  local output_nreg_thresholded_hippo_region=${23}
  local do_threshold=${24}
  local mean_brain_intensity=${25}
  local only_staple=${26}

  if [ "${only_staple}" == "no" ]
  then 
  # Brain to brain 12 dof registration
  brain_to_brain_registration ${subject_image} ${subject_brain_region} ${template_image} ${template_brain_region} \
      ${output_areg_template_brain_image} ${output_areg_template_brain_region} ${output_areg_template_brain_series_number} ${output_areg_template_air}
  regslice ${output_areg_template_air} ${template_hippo_region} ${output_areg_hippo_region} 500
  fi
  
  if [ "${only_staple}" == "no" ]
  then 
  # Hippo local rigid registration
  ${reg_template_loc} ${subject_image} ${output_areg_template_brain_image} ${subject_brain_region} ${output_areg_template_brain_region} \
      dummy ${output_areg_hippo_region}  \
      ${output_delineate_dir} ${output_delineate_dir} ${output_delineate_dir} ${output_delineate_dir} \
      ${output_study_id} ${output_series_number} ${output_echo_number} 0.5 0.8 \
      no ${output_study_id}_xor.roi 16 1 1 no 6 12
      
  # Reslice after the local 12-dof registration 
  regslice ${output_delineate_dir}/100001-${output_study_id}.air ${output_areg_hippo_region} ${output_local_areg_hippo_region} 500
  fi
  
  if [ "${only_staple}" == "no" ]
  then 
  # 3-level FFD non-rigid registration. 
  local output_local_areg_template_brain_image=${output_delineate_dir}/${output_study_id}-${output_series_number}-${output_echo_number}.img
  
  parameter_file=`mktemp /usr/tmp/param.XXXXXXXXXX`
  echo "# target image paramters      "  > ${parameter_file}
  echo "Target blurring (in mm) = 0   " >> ${parameter_file}
  echo "Target resolution (in mm) = 0 " >> ${parameter_file}
  echo "# source image paramters      " >> ${parameter_file}
  echo "Source blurring (in mm) = 0   " >> ${parameter_file}
  echo "Source resolution (in mm)  = 0" >> ${parameter_file}
  echo "# registration parameters     " >> ${parameter_file}
  echo "No. of resolution levels = 1  " >> ${parameter_file}
  echo "No. of bins = 128             " >> ${parameter_file}
  echo "No. of iterations = 20        " >> ${parameter_file}
  echo "No. of steps = 3              " >> ${parameter_file}
  echo "Length of steps = 12          " >> ${parameter_file}
  echo "Similarity measure = CC       " >> ${parameter_file}
  echo "Lambda = 0                    " >> ${parameter_file}
  echo "Interpolation mode = BSpline  " >> ${parameter_file}
  echo "Epsilon = 0.000001            " >> ${parameter_file}
  
  cat ${parameter_file}
  ${ffdnreg} ${subject_image} ${output_local_areg_template_brain_image} ${output_nreg_template_hippo_dof_16} -troi ${output_local_areg_hippo_region} \
    -dil 16 -gradient -inc 16 0 -nparams ${parameter_file}
  rm -f ${parameter_file}
  ${ffdsubdivide} ${output_nreg_template_hippo_dof_16} ${output_nreg_template_hippo_dof_16}
                    
  parameter_file=`mktemp /usr/tmp/param.XXXXXXXXXX`
  echo "# target image paramters      "  > ${parameter_file}
  echo "Target blurring (in mm) = 0   " >> ${parameter_file}
  echo "Target resolution (in mm) = 0 " >> ${parameter_file}
  echo "# source image paramters      " >> ${parameter_file}
  echo "Source blurring (in mm) = 0   " >> ${parameter_file}
  echo "Source resolution (in mm)  = 0" >> ${parameter_file}
  echo "# registration parameters     " >> ${parameter_file}
  echo "No. of resolution levels = 2  " >> ${parameter_file}
  echo "No. of bins = 128             " >> ${parameter_file}
  echo "No. of iterations = 20        " >> ${parameter_file}
  echo "No. of steps = 3              " >> ${parameter_file}
  echo "Length of steps = 6           " >> ${parameter_file}
  echo "Similarity measure = CC       " >> ${parameter_file}
  echo "Lambda = 0                    " >> ${parameter_file}
  echo "Interpolation mode = BSpline  " >> ${parameter_file}
  echo "Epsilon = 0.000001            " >> ${parameter_file}
  
  cat ${parameter_file}
  ${ffdnreg} ${subject_image} ${output_local_areg_template_brain_image} ${output_nreg_template_hippo_dof_8} -troi ${output_local_areg_hippo_region} \
    -dil 16 -gradient -inc 8 0 -nparams ${parameter_file} -inidof ${output_nreg_template_hippo_dof_16}
  rm -f ${parameter_file}
  ${ffdsubdivide} ${output_nreg_template_hippo_dof_8} ${output_nreg_template_hippo_dof_8}
  
  parameter_file=`mktemp /usr/tmp/param.XXXXXXXXXX`
  echo "# target image paramters      "  > ${parameter_file}
  echo "Target blurring (in mm) = 0   " >> ${parameter_file}
  echo "Target resolution (in mm) = 0 " >> ${parameter_file}
  echo "# source image paramters      " >> ${parameter_file}
  echo "Source blurring (in mm) = 0   " >> ${parameter_file}
  echo "Source resolution (in mm)  = 0" >> ${parameter_file}
  echo "# registration parameters     " >> ${parameter_file}
  echo "No. of resolution levels = 1  " >> ${parameter_file}
  echo "No. of bins = 128             " >> ${parameter_file}
  echo "No. of iterations = 20        " >> ${parameter_file}
  echo "No. of steps = 3              " >> ${parameter_file}
  echo "Length of steps = 3           " >> ${parameter_file}
  echo "Similarity measure = CC       " >> ${parameter_file}
  echo "Lambda = 0                    " >> ${parameter_file}
  echo "Interpolation mode = BSpline  " >> ${parameter_file}
  echo "Epsilon = 0.000001            " >> ${parameter_file}
  
  ${ffdnreg} ${subject_image} ${output_local_areg_template_brain_image} ${output_nreg_template_hippo_dof_4} -troi ${output_local_areg_hippo_region} \
    -dil 8 -gradient -inc 4 0 -nparams ${parameter_file} -inidof ${output_nreg_template_hippo_dof_8}
  
  ${ffdtransformation} ${output_local_areg_template_brain_image} ${subject_image} ${output_nreg_template_hippo_image} ${output_nreg_template_hippo_dof_4} -bspline
  ${ffdroitransformation} ${output_local_areg_hippo_region} ${output_nreg_hippo_region}  ${output_areg_template_brain_image} ${subject_image} ${output_nreg_template_hippo_dof_4} -bspline
  
  rm -f ${parameter_file}
  fi 
  
  
  # Calculate mean brain intensity
  local mean_intensity=`imginfo ${subject_image} -av -roi ${subject_brain_region}`
  if [ ${mean_brain_intensity} \> 0 ] 
  then 
    local mean_intensity=${mean_brain_intensity}
  fi 
  local threshold_70=`echo "${mean_intensity}*0.70" | bc`
  local threshold_110=`echo "${mean_intensity}*1.10" | bc`
  echo "Manual threshold=${threshold_70},${threshold_110}"
  
  temp_dir=`mktemp -d ~/temp/_hippo_mm.XXXXXX`
  local output_left_hippo_local_region_threshold_img=${temp_dir}/threshold.img
  local output_left_hippo_local_region_threshold=${temp_dir}/threshold
  local output_left_hippo_local_region_threshold_cd_img=${temp_dir}/threshold-cd.img
  local output_left_hippo_local_region_threshold_cd=${temp_dir}/threshold-cd
  
  if [ "${do_threshold}" == "both" ]
  then 
    # nreg hippo region
    makemask ${subject_image} ${output_nreg_hippo_region} ${output_left_hippo_local_region_threshold_img} -k -bpp 16
    # Threshold by 70% and 110% of mean brain intensity. 
    makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} \
      -alt ${threshold_70} -aut ${threshold_110}
    cp ${output_left_hippo_local_region_threshold} ${output_nreg_thresholded_hippo_region}
  elif [ "${do_threshold}" == "upper" ]
  then 
    # nreg hippo region
    makemask ${subject_image} ${output_nreg_hippo_region} ${output_left_hippo_local_region_threshold_img} -k -bpp 16
    # Threshold by 70% and 110% of mean brain intensity. 
    makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} \
      -aut ${threshold_110} -alt 0
    cp ${output_left_hippo_local_region_threshold} ${output_nreg_thresholded_hippo_region}
  elif [ "${do_threshold}" == "lower" ]
  then 
    # nreg hippo region
    makemask ${subject_image} ${output_nreg_hippo_region} ${output_left_hippo_local_region_threshold_img} -k -bpp 16
    # Threshold by 70% and 110% of mean brain intensity. 
    makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} \
      -alt ${threshold_70} -aut 1e10
    cp ${output_left_hippo_local_region_threshold} ${output_nreg_thresholded_hippo_region}
  else
    echo "No threshold"
    cp ${output_nreg_hippo_region} ${output_nreg_thresholded_hippo_region}
  fi 

  rm -rf ${temp_dir}
}

