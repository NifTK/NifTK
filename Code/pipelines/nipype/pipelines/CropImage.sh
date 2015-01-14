#!/bin/bash

input_image=$1
input_mask=$2
output_image=$3

# Create the cropped image
roi_dim=(`fslstats ${input_mask} -w`)

# Check if in the input image is a 5D image
dim=`nifti_tool -infiles ${input_image} -disp_nim -field ndim -quiet`
temp_image=/tmp/temp_$$_`basename ${input_image}`
if  [ ${dim} == 5 ]
then
	nt=`nifti_tool -infiles ${input_image} -disp_nim -field nt -quiet`
	if [ ${nt} -gt 1 ]
	then
		echo "Error: fslroi does not handle 5D images. Exit"
		exit
	fi
	nu=`nifti_tool -infiles ${input_image} -disp_nim -field nu -quiet`
	# Create a temp image
	cp ${input_image} ${temp_image}
	input_image=${temp_image}
	# Convert the image to 4D
	nifti_tool -infiles ${temp_image} -overwrite -mod_nim -mod_field nt ${nu} -mod_field ndim 4 -mod_field nu 1
	# Update the ROI arguments
	roi_dim="${roi_dim[0]} ${roi_dim[1]} ${roi_dim[2]} ${roi_dim[3]} ${roi_dim[4]} ${roi_dim[5]} 0 ${nu}"
	roi_dim=(${roi_dim})
fi

fslroi \
	${input_image} \
	${output_image} \
	${roi_dim[@]}

# Corrent the header information
qform_code=`nifti_tool -infiles ${input_image} -disp_nim -field qform_code -quiet`
if [ "${qform_code}" == "0" ]
then
	nifti_tool -infiles ${output_image} -overwrite -mod_nim -mod_field qform_code 1
fi
qform_matrix=(`nifti_tool -infiles ${input_image} -disp_nim -field qto_xyz -quiet`)
temp="${roi_dim[0]} ${roi_dim[2]} ${roi_dim[4]} ${qform_matrix[@]}"
qoffset_x=`echo "${temp}" | awk '{print $1 * $4 +  $2 * $5 +  $3 * $6 +  $7}'`
qoffset_y=`echo "${temp}" | awk '{print $1 * $8 +  $2 * $9 +  $3 * $10 +  $11}'`
qoffset_z=`echo "${temp}" | awk '{print $1 * $12 +  $2 * $13 +  $3 * $14 +  $15}'`
nifti_tool -infiles ${output_image} -overwrite -mod_nim \
	-mod_field qoffset_x ${qoffset_x} \
	-mod_field qoffset_y ${qoffset_y} \
	-mod_field qoffset_z ${qoffset_z}

sform_code=`nifti_tool -infiles ${input_image} -disp_nim -field sform_code -quiet`
if [ "${sform_code}" -gt "0" ]
then
	sform_matrix=(`nifti_tool -infiles ${input_image} -disp_nim -field sto_xyz -quiet`)
	temp="${roi_dim[0]} ${roi_dim[2]} ${roi_dim[4]} ${sform_matrix[@]}"
	qoffset_x=`echo "${temp}" | awk '{print $1 * $4 +  $2 * $5 +  $3 * $6 +  $7}'`
	qoffset_y=`echo "${temp}" | awk '{print $1 * $8 +  $2 * $9 +  $3 * $10 +  $11}'`
	qoffset_z=`echo "${temp}" | awk '{print $1 * $12 +  $2 * $13 +  $3 * $14 +  $15}'`
	nifti_tool -infiles ${output_image} -overwrite -mod_nim \
		-mod_field sto_xyz "${sform_matrix[0]} ${sform_matrix[1]} ${sform_matrix[2]} ${qoffset_x} \
			${sform_matrix[4]} ${sform_matrix[5]} ${sform_matrix[6]} ${qoffset_y} \
			${sform_matrix[8]} ${sform_matrix[9]} ${sform_matrix[10]} ${qoffset_z} \
			0 0 0 1"
else
	nifti_tool -infiles ${output_image} -overwrite -mod_nim \
		-mod_field sform_code 0
fi

if [ -f ${temp_image} ]
then
	nifti_tool -infiles ${output_image} -overwrite -mod_nim -mod_field nu ${nu} -mod_field ndim 5 -mod_field nt 1
	rm ${temp_image}
fi
