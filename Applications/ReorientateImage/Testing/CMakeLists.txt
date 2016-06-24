#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

add_test(niftkReorientateImage-NoOptions     ${EXECUTABLE_OUTPUT_PATH}/niftkReorientateImage -o ${TEMP}/niftkReorientateImage-NoOptions-RAI.nii ${INPUT_DATA}/reference/avg152T1_RL_nifti.nii)
add_test(niftkReorientateImage-TestNoOptions ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/niftkReorientateImage-NoOptions-RAI.nii -j ${BASELINE}/niftkReorientateImage-NoOptions-RAI.nii -intensity )
set_property(TEST niftkReorientateImage-TestNoOptions APPEND PROPERTY DEPENDS niftkReorientateImage-NoOptions)

add_test(niftkReorientateImage-OutOrient     ${EXECUTABLE_OUTPUT_PATH}/niftkReorientateImage -outOrient PRS -o ${TEMP}/niftkReorientateImage-OutOrient-PRS.nii ${INPUT_DATA}/reference/avg152T1_LR_nifti.nii)
add_test(niftkReorientateImage-TestOutOrient ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/niftkReorientateImage-OutOrient-PRS.nii -j ${BASELINE}/niftkReorientateImage-OutOrient-PRS.nii -intensity )
set_property(TEST niftkReorientateImage-TestOutOrient APPEND PROPERTY DEPENDS niftkReorientateImage-OutOrient )

add_test(niftkReorientateImage-InOutOrient     ${EXECUTABLE_OUTPUT_PATH}/niftkReorientateImage -inOrient RAS -outOrient ILP -o ${TEMP}/niftkReorientateImage-InOutOrient-ILP.nii ${INPUT_DATA}/CT_xR2L_yA2P_zS2I_Sub4.nii)
add_test(niftkReorientateImage-TestInOutOrient ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/niftkReorientateImage-InOutOrient-ILP.nii -j ${BASELINE}/niftkReorientateImage-InOutOrient-ILP.nii -intensity )
set_property(TEST niftkReorientateImage-TestInOutOrient APPEND PROPERTY DEPENDS niftkReorientateImage-InOutOrient )
