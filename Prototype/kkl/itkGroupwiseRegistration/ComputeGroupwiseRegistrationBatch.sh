#!/bin/bash 

current_dir=`dirname $0`

input_file=$1
input_dir=$2
output_dir=$3
dos2unix ${input_file}
exec 3<&0

cat ${input_file} | while read each_line
do 
  baseline_filename=`echo ${each_line} | awk -F, '{printf $1}'`.img
  baseline_basename=`basename ${baseline_filename}`
  study_id=`echo ${baseline_basename}| awk -F- '{printf $1}'`
  log=${output_dir}/${study_id}-log.txt


  qsub -l s_stack=10240 -l h_rt=72:0:0 -l vf=8G -l h_vmem=8G -S /bin/bash -j y -b y -cwd \
   -o ${log} -V ${current_dir}/ComputeGroupwiseRegistration.sh \
   30000-003-1.img 30000-003-1_dilated_mask.img 0  \
   ${each_line} ${input_dir} ${output_dir} 

#${current_dir}/ComputeGroupwiseRegistration.sh 30000-003-1.img 30000-003-1_dilated_mask.img 0 "${each_line}" ${input_dir} ${output_dir}

#exit

done 

