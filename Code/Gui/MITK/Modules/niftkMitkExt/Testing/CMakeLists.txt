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

MITK_CREATE_MODULE_TESTS(LABELS niftkMitkExt)

# Test "As Acquired" with known images, default Coronal, which is not needed here, as we should be able to work it out.
mitkAddCustomModuleTest(MIDAS-mitkAsAcquiredOrientationTest_16856   mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/16856/16856-002-1.img 2 2)
mitkAddCustomModuleTest(MIDAS-mitkAsAcquiredOrientationTest_01719-a mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-a.img 2 0)
mitkAddCustomModuleTest(MIDAS-mitkAsAcquiredOrientationTest_01719-s mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-s.img 2 1)
mitkAddCustomModuleTest(MIDAS-mitkAsAcquiredOrientationTest_01719-c mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-c.img 2 2)

# Test "As Acquired", changing default to make sure we are not needing it, output should always be sagittal.
mitkAddCustomModuleTest(MIDAS-mitkAsAcquiredOrientationTest_matt-0  mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_nifti/mdeft.1.nii 0 1)
mitkAddCustomModuleTest(MIDAS-mitkAsAcquiredOrientationTest_matt-1  mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_nifti/mdeft.1.nii 1 1)
mitkAddCustomModuleTest(MIDAS-mitkAsAcquiredOrientationTest_matt-2  mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_nifti/mdeft.1.nii 2 1)
mitkAddCustomModuleTest(MIDAS-mitkAsAcquiredOrientationTest_matt-14 mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_nifti/mdeft.1.nii 14 1)

# Test that analyze images load the same as Nifti images. Here we test intensity values, assuming same size.
mitkAddCustomModuleTest(MIDAS-mitkDRCAnalyzeSameIntensitiesAsNiftiTest_01719-a mitkMIDASCompareImagesForEqualityTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-a.img ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-a.nii 0)
mitkAddCustomModuleTest(MIDAS-mitkDRCAnalyzeSameIntensitiesAsNiftiTest_01719-s mitkMIDASCompareImagesForEqualityTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-s.img ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-s.nii 0)
mitkAddCustomModuleTest(MIDAS-mitkDRCAnalyzeSameIntensitiesAsNiftiTest_01719-c mitkMIDASCompareImagesForEqualityTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-c.img ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-c.nii 0)

# Test that if we load the same Nifti image twice, the "is equals" checks all do return true.
mitkAddCustomModuleTest(MIDAS-mitkTwoNiftiImagesAreEqualTest mitkMIDASCompareImagesForEqualityTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-a.nii ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-a.nii 0)

# Trac 1675 - Paintbrush Tool Crashing. Test single clicks with increasing cursor, scan all image, click out of bounds etc.
mitkAddCustomModuleTest(MIDAS-mitkPaintbrushToolTest mitkMIDASPaintbrushToolTest ${NIFTK_DATA_DIR}/Input/nv-11x11x11.nii)

# Trac 1683 - When nodes created via segmentation tools, the data sets much be visible in viewers
#             such as the thumbnail viewer, and the segmentation views.
mitkAddCustomModuleTest(MIDAS-mitkSegmentationNodeAddedVisibilityTest mitkMIDASSegmentationNodeAddedVisibilityTest ${NIFTK_DATA_DIR}/Input/nv-11x11x11.nii)

# Trac 1695 - Main Unit Test for the Whole MIDAS Morphological Segmentor Pipeline
mitkAddCustomModuleTest(MIDAS-mitkMIDASMorphologicalSegmentorPipelineManagerTest mitkMIDASMorphologicalSegmentorPipelineManagerTest ${NIFTK_DATA_DIR}/Input/volunteers/16856/16856-002-1.img)

# Trac 1696 - Unit tests for mitkMIDASOrientationUtils - testing things like, which axis is "up", which direction is "up", as these map to A/Z keys etc.
mitkAddCustomModuleTest(MIDAS-mitkMIDASOrientationUtilsTest-01719-a mitkMIDASOrientationUtilsTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-a.img RAI  1 -1 -1 2 0 1)
mitkAddCustomModuleTest(MIDAS-mitkMIDASOrientationUtilsTest-01719-s mitkMIDASOrientationUtilsTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-s.img ASL -1  1 -1 1 2 0)
mitkAddCustomModuleTest(MIDAS-mitkMIDASOrientationUtilsTest-01719-c mitkMIDASOrientationUtilsTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-c.img RSP -1 -1  1 1 0 2)
mitkAddCustomModuleTest(MIDAS-mitkMIDASOrientationUtilsTest-16856   mitkMIDASOrientationUtilsTest ${NIFTK_DATA_DIR}/Input/volunteers/16856/16856-002-1.img RSP -1 -1  1 1 0 2)
mitkAddCustomModuleTest(MIDAS-mitkMIDASOrientationUtilsTest-matt    mitkMIDASOrientationUtilsTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_nifti/mdeft.1.nii AIL 1 1 -1 1 2 0)

# Trac 1706 - Dilations break through connection breaker image.
mitkAddCustomModuleTest(MIDAS-mitkMIDASImageUtilsTest-16856 mitkMIDASImageUtilsTest ${NIFTK_DATA_DIR}/Input/volunteers/16856/16856-002-1.img)
