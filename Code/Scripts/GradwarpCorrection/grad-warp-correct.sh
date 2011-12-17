
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

function grad_warp_correct_image()
{
    midas_region=''
    study_id=0
    mask_file='n'
    air_file=''
    series_number='0'
    series_number_used=0
    OPTIND=1
    slices=124
    while getopts "d:s:a:b:n:u:l:m" options ; do  
        echo 'Options:' $options '=' $OPTARG 
        case $options in  
            b ) midas_region=$OPTARG;;  
            s ) study_id=$OPTARG;;  
            m ) mask_file='y';;  
            a ) air_file=$OPTARG;;  
            n ) series_number=$OPTARG;;
            u ) series_number_used=$OPTARG;;
            l ) slices=$OPTARG;;
        esac  
    done  

    echo ${study_id} " using series " ${series_number_used} "..."
    
    dicom_dirname=`mktemp -d -p /usr/tmp grad_warp_correct_image.XXXXXX` 
    no_leading_zero_study_id=`echo ${study_id} | sed 's/0*//'`
    dicom_filelist=`dicomfind.pl . -usedb -study ${study_id} -series ${series_number_used} | tail --lines=${slices} |grep "${no_leading_zero_study_id}   ${series_number_used}" ${dicom_path_file}|awk '{printf $1 " "}'` 
    number_of_dicom_files=`echo ${dicom_filelist}|wc|awk '{printf $2}'`
    echo "Number of dicom files found=${number_of_dicom_files}..."
    
    if [ ${number_of_dicom_files} \> 0 ]
    then 
        cp ${dicom_filelist} ${dicom_dirname}
        dicom_filename=${dicom_dirname}/`ls ${dicom_dirname}|head -n 1`

        if [ ${mask_file} == 'n' ]; then
            echo ${GRADUNWARP} ${dicom_dirname}/ ${study_id} "${air_file}"
            ${GRADUNWARP} ${dicom_dirname}/ ${study_id} "${air_file}"
            output_filename=`printf "%s-%03d-1.img" ${study_id} ${series_number}`
            patientId=`dcmdump $dicom_filename +P PatientID| awk -F "[" '{print $2}' |awk -F "]" '{print $1}'`
            patientName=`dcmdump $dicom_filename +P PatientsName| awk -F "[" '{print $2}' |awk -F "]" '{print $1}'`
            repetitionTime=`dcmdump $dicom_filename +P RepetitionTime| awk -F "[" '{print $2}' |awk -F "]" '{print $1}'`
            echoTime=`dcmdump $dicom_filename +P EchoTime| awk -F "[" '{print $2}' |awk -F "]" '{print $1}'`
            inversionTime=0
            inversionTime=`dcmdump $dicom_filename +P InversionTime| awk -F "[" '{print $2}' |awk -F "]" '{print $1}'`
            studyDate=`dcmdump $dicom_filename +P StudyDate| awk -F "[" '{print $2}' |awk -F "]" '{print $1}'`
            studyTime=`dcmdump $dicom_filename +P StudyTime| awk -F "[" '{print $2}' |awk -F "]" '{print $1}'`
            settime=" -time ${studyDate:6:2} ${studyDate:4:2} ${studyDate:0:4} ${studyTime:0:2} ${studyTime:2:2} "
            anchange ${study_id}_corrected.img $output_filename -cast short -study $study_id -series ${series_number} -echo 1 -mode MR -id "$patientId"  -name "$patientName" $settime -setorient cor  -tr $repetitionTime -te $echoTime -ti $inversionTime 
            if [ ${series_number} == '23' ]; then
                uncorrected_output_filename=`printf "%s-002-1.img" $study_id`
                anchange ${study_id}.img $uncorrected_output_filename -study $study_id -series 002 -echo 1 -mode MR -id "$patientId"  -name "$patientName" $settime -setorient cor  -tr $repetitionTime -te $echoTime -ti $inversionTime 
            fi 
            rm -rf ${study_id}.mgh ${study_id}_corrected.mgh ${study_id}.hdr ${study_id}.img ${study_id}_corrected.hdr ${study_id}_corrected.img
        else
            midas_image=`printf "%s-023-1.img" ${study_id}` 
            analyze_mask=${dicom_dirname}/mask
            makemask ${midas_image} ${midas_region} ${analyze_mask}.img
            echo ${GRADUNWARP} ${dicom_dirname}/ ${study_id}_region "${air_file}" ${analyze_mask}.hdr
            ${GRADUNWARP} ${dicom_dirname}/ ${study_id}_region "${air_file}" ${analyze_mask}.hdr
            makeroi -img ${study_id}_region_corrected.img -out ${study_id}_brain_region_corrected.roi -alt 127
            rm -rf ${study_id}_region.mgh ${study_id}_region_corrected.mgh ${study_id}_region.hdr ${study_id}_region.img ${study_id}_region_corrected.hdr ${study_id}_region_corrected.img
        fi
    fi

    rm -rf ${dicom_dirname} 
}



#grad_warp_correct_image -s 27523 -u 2 -n 23
#grad_warp_correct_image -s 35003 -u 3 -n 23 -m -b /var/lib/midas/data/rebecca/regions/projects/MCI2yr/Lai_35003_1208187203

#grad_warp_correct_image -s 27947 -u 2 -n 23
#grad_warp_correct_image -s 27947 -u 2 -n 23 -m -b /var/lib/midas/data/rebecca/regions/projects/hilary/Tra_27947_1106818921




































