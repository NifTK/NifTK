#!/bin/bash

set -x

##########################################################################
##########################################################################
##########################################################################

export FSLDIR=/var/drc/software/32bit/fsl
PATH=${PATH}:${FSLDIR}/bin
source ${FSLDIR}/etc/fslconf/fsl.sh
ROOTY=/var/drc/scratch1/NiftHippo
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ROOTY}/niftyreg_install/lib
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ROOTY}/nificlib_install/lib
export PATH
export LD_LIBRARY_PATH
export FSLOUTPUTTYPE=NIFTI_GZ
REG_aladin=${ROOTY}/niftyreg_install/bin/reg_aladin
REG_f3d=${ROOTY}/niftyreg_install/bin/reg_f3d
REG_resample=${ROOTY}/niftyreg_install/bin/reg_resample
REG_transform=${ROOTY}/niftyreg_install/bin/reg_transform
NIFTI_tool=${ROOTY}/nificlib_install/bin/nifti_tool
SEG_labfusion=${ROOTY}/niftyseg_install/bin/seg_LabFusion
SEG_CalcTopNCC=${ROOTY}/niftyseg_install/bin/seg_CalcTopNCC

# stddev of the Gaussian used to compute the local NCC
LNCC_GAUS=1.5 
# template folder. 
TEMPLATE_FOLDER=${ROOTY}/template_images 

##########################################################################
##########################################################################
##########################################################################
input=$1
if [ ! -e ${input} ]
then
	echo "The following image can't be found: ${input} ... Exit"
	exit
fi

maskflag=0;
# 0 equals full library (default), -jo sets library to 1
library=0; 
#Accomodate command line for template selection as well
#-jo uses jo library instead of default
#-mask maskname low and high threshold
while [ "$#" -gt 1 ]
  do
  if [ "$2" == "-mask" ]; then
      echo "Mask parameters"
      maskflag=1;
      shift
      mask_name=$2
      if [ ! -f "${mask_name}" ] 
      then 
        echo "${mask_name} not found"
        exit
      fi 
      shift
      mask_thresh_low=$2
      shift
      mask_thresh_high=$2
      shift
      echo "Name: $mask_name, Lower: $mask_thresh_low, Upper: $mask_thresh_high"
  elif [ "$2" == "-jo" ]; then
      echo "Using Jo 110 library"
      library=1;
      shift
  elif [ "$2" == "-gauss" ]; then 
      shift 
      echo "setting LNCC_GAUS=$2"
      LNCC_GAUS=$2
      shift
  elif [ "$2" == "-template" ]; then 
      shift 
      echo "setting TEMPLATE_FOLDER=$2"
      TEMPLATE_FOLDER=$2
      shift
  else
      echo "Unknown argument, options are -mask <name> <low thresh> <high thresh> and -jo"
      exit
  fi
done


if [ "$#" -gt 1 ]; then
maskflag=1;
mask_name=$2
mask_thresh_low=$3
mask_thresh_high=$4
fi


name=`basename ${input} .gz`
name=`basename ${name} .nii`
name=`basename ${name} .img`
name=`basename ${name} .hdr`

echo "NiftHippo is gald to announce that ${input} is soon to be segmented ... get ready to be amazed ... maybe"

GW_nifT1=${ROOTY}/groupwise/gw_it7_T1wBrain.nii.gz
GW_leftHippo=${ROOTY}/groupwise/gw_it7_hippo_left.nii.gz
GW_rightHippo=${ROOTY}/groupwise/gw_it7_hippo_right.nii.gz
TEMPLATE_TO_GW_TRANS_FOLDER=${ROOTY}/groupwise/nrrResult/it_7

#num_preselect=100
num_preselect=110 # dc 20/09/2011 - changed per Casper's request
LNCC_NUM=15

# Create a temporary folder
temp_folder=temp.NiftHippo.${name}.$$
if [ ! -d ${temp_folder} ]
then
	mkdir ${temp_folder}
else
	echo "The following folder already exists: ${temp_folder} ... Exit"
	exit
fi

###################################
###################################
## Extract both regions of interest
###################################
###################################

# Initial registration
###################################
${REG_aladin} \
	-target ${input} \
	-source ${GW_nifT1} \
	-aff ${temp_folder}/gw_to_${name}_affine_mat.txt \
	-result ${temp_folder}/gw_to_${name}_affine_res.nii.gz
if [ ! -e ${temp_folder}/gw_to_${name}_affine_mat.txt ]
then
	echo "[NiftHippo ERROR] Affine from groupwise to input image"
	exit
fi
${REG_f3d} \
	-target ${input} \
	-source ${GW_nifT1} \
	-aff ${temp_folder}/gw_to_${name}_affine_mat.txt \
	-result ${temp_folder}/gw_to_${name}_nrr_res.nii.gz \
	-cpp ${temp_folder}/gw_to_${name}_nrr_cpp.nii.gz \
	-lp 2 -voff
if [ ! -e ${temp_folder}/gw_to_${name}_nrr_cpp.nii.gz  ]
then
	echo "[NiftHippo ERROR] Non-rigid from groupwise to input image"
	exit
fi
###################################
for side in left right
  do
	# Define the ROI
  sourceImage=${GW_leftHippo}
  if [ "${side}" == "right" ]; then sourceImage=${GW_rightHippo};fi
  ${REG_resample} \
      -target ${input} \
      -source ${sourceImage} \
      -cpp ${temp_folder}/gw_to_${name}_nrr_cpp.nii.gz \
      -TRI \
      -result ${temp_folder}/${side}_hippo_roi_${name}.nii.gz
  if [ ! -e ${temp_folder}/${side}_hippo_roi_${name}.nii.gz ]
      then
      echo "[NiftHippo ERROR] ROI extraction"
      exit
  fi
  
  cp ${temp_folder}/${side}_hippo_roi_${name}.nii.gz ${temp_folder}/${side}_hippo_binseg_${name}.nii.gz
  
  fslmaths \
      ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
      -s 1.5 -bin \
      ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
      -odt char
	###########################
	# Extract the ROI
  ROI=(`fslstats ${temp_folder}/${side}_hippo_roi_${name}.nii.gz -w`)
  
  fslroi \
      ${temp_folder}/${side}_hippo_binseg_${name}.nii.gz \
      ${temp_folder}/${side}_hippo_binseg_${name}.nii.gz \
      ${ROI[@]}
  
  fslroi \
      ${input} \
      ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
      ${ROI[@]}
	###########################
	# Change the origin - Qform
  qto_xyz=(`${NIFTI_tool} -disp_nim -infiles ${input} -field qto_xyz -quiet`)
  origin_x=`echo "${ROI[0]} ${ROI[2]} ${ROI[4]} ${qto_xyz[@]}" | awk '{print $1 * $4  + $2 * $5  + $3 * $6  + $7}'`
  origin_y=`echo "${ROI[0]} ${ROI[2]} ${ROI[4]} ${qto_xyz[@]}" | awk '{print $1 * $8  + $2 * $9  + $3 * $10 + $11}'`
  origin_z=`echo "${ROI[0]} ${ROI[2]} ${ROI[4]} ${qto_xyz[@]}" | awk '{print $1 * $12 + $2 * $13 + $3 * $14 + $15}'`
  ${NIFTI_tool} \
      -infiles ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
      -mod_nim \
      -overwrite \
      -mod_field qform_code 1 \
      -mod_field sform_code 0 \
      -mod_field qoffset_x ${origin_x} \
      -mod_field qoffset_y ${origin_y} \
      -mod_field qoffset_z ${origin_z}
	###########################
	# Change the origin - Sform if defined
  sform_code=`${NIFTI_tool} -disp_nim -infiles ${input} -field sform_code -quiet`
  if [ "${sform_code}" -gt 0 ]
      then
      sto_xyz=(`${NIFTI_tool} -disp_nim -infiles ${input} -field sto_xyz -quiet`)
      origin_x=`echo "${ROI[0]} ${ROI[2]} ${ROI[4]} ${sto_xyz[@]}" | awk '{print $1 * $4  + $2 * $5  + $3 * $6  + $7}'`
      origin_y=`echo "${ROI[0]} ${ROI[2]} ${ROI[4]} ${sto_xyz[@]}" | awk '{print $1 * $8  + $2 * $9  + $3 * $10 + $11}'`
      origin_z=`echo "${ROI[0]} ${ROI[2]} ${ROI[4]} ${sto_xyz[@]}" | awk '{print $1 * $12 + $2 * $13 + $3 * $14 + $15}'`
      sto_xyz=`echo "${sto_xyz[@]} ${origin_x} ${origin_y} ${origin_z}" | awk '{print $1 " " $2 " " $3 " " $17 " " $5 " " $6 " " $7 " " $18 " " $9 " " $10 " " $11 " " $19 " 0 0 0 1"}'`
      ${NIFTI_tool} \
	  -infiles ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
	  -mod_nim \
	  -overwrite \
	  -mod_field sform_code 1 \
	  -mod_field sto_xyz "${sto_xyz}"
  fi
	###########################
done
###################################

###########################################
###########################################
## Propagate all templates to the new image
###########################################
###########################################

for side in left right
  do
#if statement starts here, IF regular NIftHIppo, include all of this
  if [ "$library" -eq 0 ]; then
	###################################
	# Registration from groupwise to subject in the ROI
      ${REG_aladin} \
	  -inaff ${temp_folder}/gw_to_${name}_affine_mat.txt \
	  -target ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
	  -source ${GW_nifT1} \
	  -aff ${temp_folder}/gw_to_${name}_${side}_affine_mat.txt \
	  -result ${temp_folder}/gw_to_${name}_${side}_affine_res.nii.gz
      if [ ! -e ${temp_folder}/gw_to_${name}_${side}_affine_mat.txt ]
	  then
	  echo "[NiftHippo ERROR] Affine groupwise to input ROI"
	  exit
      fi
      ${REG_f3d} \
	  -target ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
	  -source ${GW_nifT1} \
	  -aff ${temp_folder}/gw_to_${name}_${side}_affine_mat.txt \
	  -result ${temp_folder}/gw_to_${name}_${side}_nrr_res.nii.gz \
	  -be 0.01 \
	  -cpp ${temp_folder}/gw_to_${name}_${side}_nrr_cpp.nii.gz -voff
      if [ ! -e ${temp_folder}/gw_to_${name}_${side}_nrr_cpp.nii.gz ]
	  then
	  echo "[NiftHippo ERROR] Non-rigid groupwise to input ROI"
	  exit
      fi
      ${REG_transform} \
	  -target ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
	  -cpp2def \
	  ${temp_folder}/gw_to_${name}_${side}_nrr_cpp.nii.gz \
	  ${temp_folder}/gw_to_${name}_${side}_nrr_def.nii.gz
      if [ ! -e ${temp_folder}/gw_to_${name}_${side}_nrr_def.nii.gz ]
	  then
	  echo "[NiftHippo ERROR] Deformation field from groupwise to input ROI"
	  exit
      fi
	###################################
	# List of all available templates
      warpedTemplateList=""
      templateNumber=0
	###################################
      template_list=""
      if [ "${side}" == "left" ]; then
	  template_list=`ls ${TEMPLATE_FOLDER}/updatedSform_hippo_left_[0-9]*-1.nii.gz  ${TEMPLATE_FOLDER}/updatedSform_hippo_right_[0-9]*-1_LR_flipped.nii.gz`
      else
	  template_list=`ls ${TEMPLATE_FOLDER}/updatedSform_hippo_right_[0-9]*-1.nii.gz  ${TEMPLATE_FOLDER}/updatedSform_hippo_left_[0-9]*-1_LR_flipped.nii.gz`
      fi

      for template in ${template_list}
	do
	###########################
	# Extract the name of the T1w corresponding image
	template_name=`basename ${template} .nii.gz`
	hippo_side=`echo "${template_name}" | awk -F "_" '{print $3}'`
	template_name=`echo "${template_name}" | awk -F "t_" '{print $2}'`
	###########################
		# Create the deformation field
	${REG_transform} \
	    -target ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
	    -comp2 \
	    ${TEMPLATE_TO_GW_TRANS_FOLDER}/c-cpp_${template_name}it-7.nii.gz \
	    ${temp_folder}/gw_to_${name}_${side}_nrr_def.nii.gz \
	    ${temp_folder}/tempDef.nii.gz
		##########################
		# Deforme the template in the space of the input image
	${REG_resample} \
	    -target ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
	    -source ${ROOTY}/nii_image_and_flipped/${template_name}.nii.gz \
	    -def ${temp_folder}/tempDef.nii.gz \
	    -result ${temp_folder}/gw_warped_${hippo_side}_${template_name}.nii.gz

	if [ ! -e ${temp_folder}/gw_warped_${hippo_side}_${template_name}.nii.gz ]
	    then
	    echo "[NiftHippo ERROR] Template resampling to input ROI"
	    exit
	fi
		###########################
		# Store the warped image name
	warpedTemplateList="${warpedTemplateList} ${temp_folder}/gw_warped_${hippo_side}_${template_name}.nii.gz"
	templateNumber=`expr ${templateNumber} + 1`
      done
	###################################
      
      best_templates=`${SEG_CalcTopNCC} \
                        -target ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
                        -templates ${templateNumber} ${warpedTemplateList} \
                        -n ${num_preselect}\
                        -mask ${temp_folder}/${side}_hippo_binseg_${name}.nii.gz`
  else
#else just define the 110 images in best_templates
      jo_codes=( 30026 30027 30028 30030 30038 30041 30043 30058 30064 30071 30073 30096 30114 30117 30119 30121 30123 30148 30150 30153 30157 30159 30163 30182 30187 30190 30227 30233 30240 30249 30250 30252 30265 30266 30282 30314 30317 30336 30339 30369 30372 30408 30434 30441 30444 30458 30459 30472 30488 30489 30520 30526 30528 30611 30614 )
      best_templates=""
      for j in ${jo_codes[*]}
	do
	if [ "${side}" == "left" ]; then
	    best_templates="${best_templates} ${TEMPLATE_FOLDER}/updatedSform_hippo_left_${j}-333-1.nii.gz ${TEMPLATE_FOLDER}/updatedSform_hippo_right_${j}-333-1_LR_flipped.nii.gz"
	else
	    best_templates="${best_templates} ${TEMPLATE_FOLDER}/updatedSform_hippo_right_${j}-333-1.nii.gz ${TEMPLATE_FOLDER}/updatedSform_hippo_left_${j}-333-1_LR_flipped.nii.gz"
	fi
      done
      
  fi
	##################################
  warpedTemplateList=""
  warpedSegmentationList=""
  for template in ${best_templates}
    do
		###########################
    template_name=`basename ${template} .nii.gz`
    hippo_side=`echo "${template_name}" | awk -F "_" '{print $3}'`
    template_name=`echo "${template_name}" | awk -F "t_" '{print $2}'`
    echo "$template $template_name"
    source=${TEMPLATE_FOLDER}/updatedSform_${template_name}.nii.gz
    hyppo=${TEMPLATE_FOLDER}/updatedSform_hippo_${hippo_side}_${template_name}.nii.gz
	
		###########################
		# Registration from template to subject
    ${REG_aladin} \
	-target ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
	-inaff ${temp_folder}/gw_to_${name}_affine_mat.txt \
	-source ${source} \
	-aff ${temp_folder}/${template_name}_to_${name}_${side}_affine_mat.txt \
	-result ${temp_folder}/${template_name}_to_${name}_${side}_affine_res.nii.gz
    if [ ! -e ${temp_folder}/${template_name}_to_${name}_${side}_affine_mat.txt ]
	then
	echo "[NiftHippo ERROR] Affine template to input ROI"
	exit
    fi
    ${REG_f3d} \
	-target ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
	-source ${source} \
	-aff ${temp_folder}/${template_name}_to_${name}_${side}_affine_mat.txt \
	-result ${temp_folder}/${template_name}_to_${name}_${side}_nrr_res.nii.gz \
	-cpp ${temp_folder}/${template_name}_to_${name}_${side}_nrr_cpp.nii.gz \
	-sx -2.5 \
	-be 0.01 -voff
    if [ ! -e ${temp_folder}/${template_name}_to_${name}_${side}_nrr_cpp.nii.gz ]
	then
	echo "[NiftHippo ERROR] Non-rigid template to input ROI"
	exit
    fi
    ${REG_resample} \
	-target ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
	-source ${hyppo} \
	-cpp ${temp_folder}/${template_name}_to_${name}_${side}_nrr_cpp.nii.gz \
	-result ${temp_folder}/${template_name}_to_${name}_${side}_seg.nii.gz \
	-NN
    if [ ! -e ${temp_folder}/${template_name}_to_${name}_${side}_seg.nii.gz ]
	then
	echo "[NiftHippo ERROR] Segmentation propagation from template to input ROI"
	exit
    fi
    warpedTemplateList="${warpedTemplateList} ${temp_folder}/${template_name}_to_${name}_${side}_nrr_res.nii.gz "
    warpedSegmentationList="${warpedSegmentationList} ${temp_folder}/${template_name}_to_${name}_${side}_seg.nii.gz "
  done
	#################################
  fslmerge -t ${temp_folder}/${side}_allTemplates_${name}.nii.gz ${warpedTemplateList}
  fslmerge -t ${temp_folder}/${side}_allSegmentation_${name}.nii.gz ${warpedSegmentationList}
  ${SEG_labfusion} \
      -in  ${temp_folder}/${side}_allSegmentation_${name}.nii.gz \
      -unc \
      -STAPLE \
      -out ${temp_folder}/${side}_hippo_seg_roi_${name}.nii.gz \
      -LNCC \
      ${LNCC_GAUS} \
      ${LNCC_NUM} \
      ${temp_folder}/${side}_hippo_roi_${name}.nii.gz \
      ${temp_folder}/${side}_allTemplates_${name}.nii.gz \
      -MRF_beta 0.55 #dc 20/09/2011 - Changed as it was somehow deleted.
  if [ ! -e ${temp_folder}/${side}_allTemplates_${name}.nii.gz ]
      then
      echo "[NiftHippo ERROR] Fusion"
      exit
  fi
	###################################
  ${REG_resample} \
      -target ${input} \
      -source ${temp_folder}/${side}_hippo_seg_roi_${name}.nii.gz \
      -result ${name}_${side}_NiftHippo.nii.gz \
      -NN
  if [ ! -e ${name}_${side}_NiftHippo.nii.gz ]
      then
      echo "[NiftHippo ERROR] final resampling in original input image"
      exit
  fi
  
  if [ $maskflag -eq 1 ]; then
      mean_intensity=`fslstats ${input} -k ${mask_name} -m`
      intensity_thresh_high=`echo "scale=3; ${mean_intensity} * ${mask_thresh_high} / 100" | bc`
      intensity_thresh_low=`echo "scale=3; ${mean_intensity} * ${mask_thresh_low} / 100" | bc`
      fslmaths \
	  $input \
	  -thr ${intensity_thresh_low} \
	  -uthr ${intensity_thresh_high} \
	  -bin \
	  -mul ${name}_${side}_NiftHippo.nii.gz \
	  ${name}_${side}_masked_NiftHippo.nii.gz
      
      if [ ! -e ${name}_${side}_masked_NiftHippo.nii.gz ]
	  then
	  echo "[NiftHippo ERROR] final resampling of the masked hippo in original input image"
	  exit
      fi
  fi
  
  
	##################################
done
###########################################
rm -rf ${temp_folder}
###########################################
