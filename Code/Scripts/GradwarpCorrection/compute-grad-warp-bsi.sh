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

input_file=$1
reg_dir=$2

if [ $# \< 2 ]
then
  echo "Program to perform grad warp correction and calculate BSI"
  echo "Usage: $0 input_csv_file output_dir"
  echo "    where input_csv_file has the format:"
  echo "        baseline_study_id,baseline_series_no,baseline_number_of_slices,baseline_region,repeat_study_id,repeat_series_no,repeat_number_of_slicesrepeat_region"
  echo "        e.g. 26599,2,124,/var/lib/midas/data/rebecca/regions/projects/hilary/liz_26599_1093964501,35612,2,124,/var/lib/midas/data/rebecca/regions/projects/MCI2yr/Lai_35612_1208785727"
  echo ""  
  echo "Requirments: GRAD_UNWARP_DIR: install directory of grad warp correction package"
  echo "               e.g. export GRAD_UNWARP_DIR=/var/tmp/work/GradUnwarp" 
  echo "             MATLAB_PATH: MATLAB binary location"
  echo "               e.g. export MATLAB_PATH=/usr/local/lib/matlab-r14/bin"
  exit 0
fi

# for running c++ compiled code in slackware matlab installation. 
export LD_PRELOAD=/usr/lib/libgcc_s.so.1

export GRADUNWARP=${GRAD_UNWARP_DIR}/unwarp.sh

source `dirname "$0"`/grad-warp-correct.sh

dos2unix ${input_file}

cat ${input_file} | while read each_line 
do
    tmpdir=`mktemp -d -q /usr/tmp/midasreg.XXXXXX`

    baseline_study_id=`echo ${each_line} | awk -F, '{printf $1}'`
    baseline_series=`echo ${each_line} | awk -F, '{printf $2}'`
    baseline_slcies=`echo ${each_line} | awk -F, '{printf $3}'`
    baseline_region=`echo ${each_line} | awk -F, '{printf $4}'`
    repeat_study_id=`echo ${each_line} | awk -F, '{printf $5}'`
    repeat_series=`echo ${each_line} | awk -F, '{printf $6}'`
    repeat_slices=`echo ${each_line} | awk -F, '{printf $7}'`
    repeat_region=`echo ${each_line} | awk -F, '{printf $8}'`
    baseline_study_id=`printf "%05d" ${baseline_study_id}`
    repeat_study_id=`printf "%05d" ${repeat_study_id}`
    bsi_output_file=${baseline_study_id}-${repeat_study_id}.qnt
    echo "Processing ${each_line}..."
    
    if [ ! -f ${bsi_output_file} ] && [ ! -z ${baseline_study_id} ] && [ ! -z ${baseline_series} ] && [ ! -z ${baseline_region} ] && [ ! -z ${repeat_study_id} ] && [ ! -z ${repeat_series} ] && [ ! -z ${repeat_region} ]
    then 

        grad_warp_correct_image -s ${repeat_study_id} -u ${repeat_series} -n 23 -l ${repeat_slices}
        grad_warp_correct_image -s ${repeat_study_id} -u ${repeat_series} -m -l ${repeat_slices} -b ${repeat_region}
        grad_warp_correct_image -s ${baseline_study_id} -u ${baseline_series} -n 23 -l ${baseline_slcies}
        grad_warp_correct_image -s ${baseline_study_id} -u ${baseline_series} -m -l ${baseline_slcies} -b ${baseline_region}

        baseline_image=${baseline_study_id}-023-1.img
        repeat_image=${repeat_study_id}-023-1.img
        baseline_roi=${baseline_study_id}_brain_region_corrected.roi
        repeat_roi=${repeat_study_id}_brain_region_corrected.roi
        
        if [ -f ${baseline_image} ] && [ -f ${repeat_image} ] && [ -f ${baseline_roi} ] && [ -f ${repeat_roi} ]
        then 

            baseline_analyze_roi=${tmpdir}/smask.img
            repeat_analyze_roi=${tmpdir}/rmask.img
            prealign_air=${tmpdir}/pre.air
            align_air=${tmpdir}/${baseline_study_id}-${repeat_study_id}-i.air
            align_air_ini=${tmpdir}/${baseline_study_id}-${repeat_study_id}-i.ini
            final_align_air=${baseline_study_id}-${repeat_study_id}.air
            final_align_air_matrix=${baseline_study_id}-${repeat_study_id}.txt
            resliced_repeat_image_prefix=${repeat_study_id}-007-1
            resliced_repeat_roi=${repeat_study_id}-007-1.roi
            bias_corrected_baseline_image_prefix=${baseline_study_id}-012-1
            bias_corrected_repeat_image_prefix=${repeat_study_id}-012-1
            bias_correct_baseline_script=bias_correct_baseline_script_${baseline_study_id}.csh
            bias_correct_repeat_script=bias_correct_repeat_script_${repeat_study_id}.csh
            bias_corrected_baseline_image_dbc=${repeat_study_id}-048-1
            bias_corrected_repeat_image_dbc=${repeat_study_id}-049-1

            # Registration degrees of freedom
            dof_reg=12

            $MIDAS_BIN/makemask ${baseline_image} ${baseline_roi} ${baseline_analyze_roi} 
            $MIDAS_BIN/makemask ${repeat_image} ${repeat_roi} ${repeat_analyze_roi} 

            # Performing pre-alignment calculation...
            $MIDAS_BIN/reg_prealign ${baseline_image} ${repeat_image} ${baseline_roi} ${repeat_roi} ${prealign_air} -t1 0.2 -t2 0.2 -a -$dof_reg

            t1=`$MIDAS_BIN/imginfo ${baseline_image} -tanz 0.2`
            t2=`$MIDAS_BIN/imginfo ${repeat_image} -tanz 0.2`

            # AIR registration. 
            rm -f ${align_air}
            $AIR_BIN/alignlinear ${baseline_image} ${repeat_image} ${align_air} -m $dof_reg -e1 ${baseline_analyze_roi} -e2 ${repeat_analyze_roi} -f ${prealign_air} -g ${align_air_ini} y -p1 1 -p2 1 -s 81 1 3 -c 0.000001 -h 200 -r 200 -q -x 1 -t1 $t1 -t2 $t2 -v -b1 2.1876 2.1876 3.000 -b2 2.1876 2.1876 3.000
            rm -f ${final_align_air}
            $AIR_BIN/alignlinear ${baseline_image} ${repeat_image} ${final_align_air} -m $dof_reg -e1 ${baseline_analyze_roi} -e2 ${repeat_analyze_roi} -f ${align_air_ini} -p1 1 -p2 1 -s 2 1 2 -c 0.0000001 -h 200 -r 200 -q -x 1 -t1 $t1 -t2 $t2 -v 

            rm -f ${resliced_repeat_image_prefix}.img ${resliced_repeat_image_prefix}.hdr 

            # Dump the transformation matrix from the AIR file. 
            rm -f ${final_align_air_matrix}
            air2fsl ${final_align_air} ${final_align_air_matrix}

            # Reslcie the repeat image. 
            grad_warp_correct_image -s ${repeat_study_id} -u ${repeat_series} -n 7 -a ${final_align_air_matrix}
            # Relice the repeat region. 
            $MIDAS_BIN/regslice ${final_align_air} ${repeat_roi} ${resliced_repeat_roi} 7 -3 -i 2  -c 

            # Bias correction. 
            $MIDAS_BIN/bias_correct ${bias_correct_baseline_script} ${baseline_image} ${baseline_roi}
            $MIDAS_BIN/bias_correct ${bias_correct_repeat_script} ${resliced_repeat_image_prefix}.img ${resliced_repeat_roi}
            ./${bias_correct_baseline_script}
            ./${bias_correct_repeat_script}

            $MIDAS_BIN/differentialbiascorrect ${bias_corrected_baseline_image_prefix} ${bias_corrected_repeat_image_prefix}  ${baseline_roi} ${resliced_repeat_roi} `pwd` 5 ${tmpdir} ${bias_corrected_baseline_image_dbc} ${bias_corrected_repeat_image_dbc} 2 0 0 0 0

            ${MIDAS_BIN}/extend_header ${bias_corrected_baseline_image_dbc}.img ${baseline_image} ${reg_dir} 48 1 ${repeat_image:0:5} -title
            ${MIDAS_BIN}/extend_header ${bias_corrected_repeat_image_dbc}.img ${repeat_image} ${reg_dir} 49 1 ${repeat_image:0:5} -title

            # BSI calculation. 
            $MIDAS_BIN/bsi_calc ${final_align_air} ${baseline_roi} ${reg_dir}/${bias_corrected_baseline_image_dbc}.img ${repeat_roi} ${repeat_image}  ${reg_dir}/${bias_corrected_repeat_image_dbc}.img -3 -e 1 -d 1  -l 0.45 -u 0.75  -t 0.0 -c 0 > ${bsi_output_file}

            #gzip -f ${baseline_study_id}*.img ${repeat_study_id}*.img ${reg_dir}/${baseline_study_id}*.img ${reg_dir}/${repeat_study_id}*.img
        fi
    fi
    rm -rf $tmpdir
    
done











