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

# Example test, documented in NifTK/Doc/TechnicalManual/TestingApplications.dox

add_test(niftkAdd-Run2D ${EXECUTABLE_OUTPUT_PATH}/niftkAdd -i ${INPUT_DATA}/cte_circle_gm.png -j ${INPUT_DATA}/cte_circle_wm.png -o ${TEMP}/niftkAdd-2D-output.png)
add_test(niftkAdd-Test2D ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/niftkAdd-2D-output.png -j ${BASELINE}/niftkAdd-2D-output.png -intensity )

# Additional image checks can be added to niftkTestCompareImage to make it more useful.
# Additional ctest tests should be added below, eventually covering all the applications in NifTK/Code/Applications.

# niftkReorientateImage

add_test(niftkReorientateImage-NoOptions     ${EXECUTABLE_OUTPUT_PATH}/niftkReorientateImage -o ${TEMP}/niftkReorientateImage-NoOptions-RAI.nii ${INPUT_DATA}/reference/avg152T1_RL_nifti.nii)
add_test(niftkReorientateImage-TestNoOptions ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/niftkReorientateImage-NoOptions-RAI.nii -j ${BASELINE}/niftkReorientateImage-NoOptions-RAI.nii -intensity )

add_test(niftkReorientateImage-OutOrient     ${EXECUTABLE_OUTPUT_PATH}/niftkReorientateImage -outOrient PRS -o ${TEMP}/niftkReorientateImage-OutOrient-PRS.nii ${INPUT_DATA}/reference/avg152T1_LR_nifti.nii)
add_test(niftkReorientateImage-TestOutOrient ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/niftkReorientateImage-OutOrient-PRS.nii -j ${BASELINE}/niftkReorientateImage-OutOrient-PRS.nii -intensity )

add_test(niftkReorientateImage-InOutOrient     ${EXECUTABLE_OUTPUT_PATH}/niftkReorientateImage -inOrient RAS -outOrient ILP -o ${TEMP}/niftkReorientateImage-InOutOrient-ILP.nii ${INPUT_DATA}/CT_xR2L_yA2P_zS2I_Sub4.nii)
add_test(niftkReorientateImage-TestInOutOrient ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/niftkReorientateImage-InOutOrient-ILP.nii -j ${BASELINE}/niftkReorientateImage-InOutOrient-ILP.nii -intensity )

# niftkOtsuThresholdImage

add_test(niftkOtsuThresholdImage-NoiseySquare2D ${EXECUTABLE_OUTPUT_PATH}/niftkOtsuThresholdImage -i ${INPUT_DATA}/NoiseySquare2D.nii.gz -o ${TEMP}/OtsuThresholdMaskNoiseySquare2D.nii)
add_test(niftkOtsuThresholdImage-TestNoiseySquare2D ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/OtsuThresholdMaskNoiseySquare2D.nii -j ${BASELINE}/OtsuThresholdMaskNoiseySquare2D.nii -intensity )

add_test(niftkOtsuThresholdImage-NoiseySquare2DInvert ${EXECUTABLE_OUTPUT_PATH}/niftkOtsuThresholdImage --invert -i ${INPUT_DATA}/NoiseySquare2D.nii.gz -o ${TEMP}/InvertedOtsuThresholdMaskNoiseySquare2D.nii)
add_test(niftkOtsuThresholdImage-TestNoiseySquare2DInvert ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/InvertedOtsuThresholdMaskNoiseySquare2D.nii -j ${BASELINE}/InvertedOtsuThresholdMaskNoiseySquare2D.nii -intensity )

add_test(niftkOtsuThresholdImage-NoiseyCuboid3D ${EXECUTABLE_OUTPUT_PATH}/niftkOtsuThresholdImage -i ${INPUT_DATA}/NoiseyCuboid3D.nii.gz -o ${TEMP}/OtsuThresholdMaskNoiseyCuboid3D.nii)
add_test(niftkOtsuThresholdImage-TestNoiseyCuboid3D ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/OtsuThresholdMaskNoiseyCuboid3D.nii -j ${BASELINE}/OtsuThresholdMaskNoiseyCuboid3D.nii -intensity )

add_test(niftkOtsuThresholdImage-NoiseyCuboid3DInvert ${EXECUTABLE_OUTPUT_PATH}/niftkOtsuThresholdImage --invert -i ${INPUT_DATA}/NoiseyCuboid3D.nii.gz -o ${TEMP}/InvertedOtsuThresholdMaskNoiseyCuboid3D.nii)
add_test(niftkOtsuThresholdImage-TestNoiseyCuboid3DInvert ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/InvertedOtsuThresholdMaskNoiseyCuboid3D.nii -j ${BASELINE}/InvertedOtsuThresholdMaskNoiseyCuboid3D.nii -intensity )
