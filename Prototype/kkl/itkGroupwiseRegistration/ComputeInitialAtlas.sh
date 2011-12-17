#!/bin/bash

current_dir=`dirname $0`

  exe_path=${current_dir}/../../../../NifTK-build/bin/

atlas=$1
output_format=$2
fixed_image=$3
fixed_image_mask=$4
resampling=$5

if [ ! -f ${fixed_image} ]
then 
  echo ${fixed_image}" not found!"
  exit
fi

if [ ! -f ${fixed_image_mask} ]
then 
  echo ${fixed_image_mask}" not found!"
  exit
fi


is_resample=`echo " ${resampling} > 0"|bc`

moving_image_and_dof=""
resample_flag=""

if [ ${is_resample} != 0 ]
then
  resample_flag="-resample ${resampling}"
fi  
fixed_image_resampled=${fixed_image}
fixed_image_mask_resampled=${fixed_image_mask}
temp_dofs=""

shift 5
i=0
while [ ! -z "$1" ]
do
  output=`printf ${output_format} ${i}`
  moving_image=$1
  if [ ! -f ${moving_image} ]
  then 
    echo ${moving_image}" not found!"
    exit
  fi
  
  moving_image_resampled=${moving_image}

  ${exe_path}niftkFluid \
     -ti ${fixed_image_resampled} -si ${moving_image_resampled} \
     -tm ${fixed_image_mask_resampled} \
     -oi ${output}.hdr \
     -to ${output}.dof \
     -ln 1 -fi 4 -ri 2 -is 0.7 -cs 1 -md 0.05 -mc 1.0e-9 -ls 1.0 -rs 1.0 -force ssdn -sim 4 ${resample_flag} \
     -hfl -1e6 -hfu 1e6 -hml -1e6 -hmu 1e6
   
  moving_image_and_dof=${moving_image_and_dof}" ${moving_image_resampled} ${output}.dof"
  temp_dofs="${temp_dofs} ${output}.dof"
     
  shift 1 
  (( i++ ))
done

${exe_path}itkComputeInitialAtlas ${atlas} ${output_format}-nreg-init.dof ${fixed_image_resampled} 4 ${moving_image_and_dof}

dofin=`printf ${output_format}-nreg-init.dof 0`
${exe_path}niftkTransformation -ti ${fixed_image_mask_resampled} -si ${fixed_image_mask_resampled} \
                                -o ${fixed_image_mask_resampled%.*}-nreg.hdr \
                                -df ${dofin} \
                                -j 1 
                                
                                
# Tidy up dofs. 
rm -f ${temp_dofs}                                

#spacing=10
# Not working with FFD for now. 
#/home/samba/user/leung/work/NifTK-build/bin/niftkFFD \
#     -ti ${fixed_image} -si $1 \
#     -oi ${output}.hdr \
#     -ot ${output}.dof \
#     -sx ${spacing} -sy ${spacing} -sz ${spacing} \
#     -ln 1 -fi 4 -gi 4 -ri 2 
      




