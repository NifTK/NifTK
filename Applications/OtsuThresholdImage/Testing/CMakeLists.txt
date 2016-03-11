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

add_test(niftkOtsuThresholdImage-NoiseySquare2D ${EXECUTABLE_OUTPUT_PATH}/niftkOtsuThresholdImage -i ${INPUT_DATA}/NoiseySquare2D.nii.gz -o ${TEMP}/OtsuThresholdMaskNoiseySquare2D.nii)
add_test(niftkOtsuThresholdImage-TestNoiseySquare2D ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/OtsuThresholdMaskNoiseySquare2D.nii -j ${BASELINE}/OtsuThresholdMaskNoiseySquare2D.nii -intensity )
set_property(TEST niftkOtsuThresholdImage-TestNoiseySquare2D APPEND PROPERTY DEPENDS niftkOtsuThresholdImage-NoiseySquare2D )

add_test(niftkOtsuThresholdImage-NoiseySquare2DInvert ${EXECUTABLE_OUTPUT_PATH}/niftkOtsuThresholdImage --invert -i ${INPUT_DATA}/NoiseySquare2D.nii.gz -o ${TEMP}/InvertedOtsuThresholdMaskNoiseySquare2D.nii)
add_test(niftkOtsuThresholdImage-TestNoiseySquare2DInvert ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/InvertedOtsuThresholdMaskNoiseySquare2D.nii -j ${BASELINE}/InvertedOtsuThresholdMaskNoiseySquare2D.nii -intensity )
set_property(TEST niftkOtsuThresholdImage-TestNoiseySquare2DInvert APPEND PROPERTY DEPENDS niftkOtsuThresholdImage-NoiseySquare2DInvert )

add_test(niftkOtsuThresholdImage-NoiseyCuboid3D ${EXECUTABLE_OUTPUT_PATH}/niftkOtsuThresholdImage -i ${INPUT_DATA}/NoiseyCuboid3D.nii.gz -o ${TEMP}/OtsuThresholdMaskNoiseyCuboid3D.nii)
add_test(niftkOtsuThresholdImage-TestNoiseyCuboid3D ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/OtsuThresholdMaskNoiseyCuboid3D.nii -j ${BASELINE}/OtsuThresholdMaskNoiseyCuboid3D.nii -intensity )
set_property(TEST niftkOtsuThresholdImage-TestNoiseyCuboid3D APPEND PROPERTY DEPENDS niftkOtsuThresholdImage-NoiseyCuboid3D )

add_test(niftkOtsuThresholdImage-NoiseyCuboid3DInvert ${EXECUTABLE_OUTPUT_PATH}/niftkOtsuThresholdImage --invert -i ${INPUT_DATA}/NoiseyCuboid3D.nii.gz -o ${TEMP}/InvertedOtsuThresholdMaskNoiseyCuboid3D.nii)
add_test(niftkOtsuThresholdImage-TestNoiseyCuboid3DInvert ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/InvertedOtsuThresholdMaskNoiseyCuboid3D.nii -j ${BASELINE}/InvertedOtsuThresholdMaskNoiseyCuboid3D.nii -intensity )
set_property(TEST niftkOtsuThresholdImage-TestNoiseyCuboid3DInvert APPEND PROPERTY DEPENDS niftkOtsuThresholdImage-NoiseyCuboid3DInvert )
