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

MITK_CREATE_MODULE_TESTS(LABELS niftkMIDAS)

# Trac 1675 - Paintbrush Tool Crashing. Test single clicks with increasing cursor, scan all image, click out of bounds etc.
mitkAddCustomModuleTest(MIDAS-mitkPaintbrushToolTest mitkMIDASPaintbrushToolTest ${NIFTK_DATA_DIR}/Input/nv-11x11x11.nii)

# Trac 1683 - When nodes created via segmentation tools, the data sets much be visible in viewers
#             such as the thumbnail viewer, and the segmentation views.
mitkAddCustomModuleTest(MIDAS-mitkSegmentationNodeAddedVisibilityTest mitkMIDASSegmentationNodeAddedVisibilityTest ${NIFTK_DATA_DIR}/Input/nv-11x11x11.nii)

# Trac 1695 - Main Unit Test for the Whole MIDAS Morphological Segmentor Pipeline
mitkAddCustomModuleTest(MIDAS-mitkMIDASMorphologicalSegmentorPipelineManagerTest mitkMIDASMorphologicalSegmentorPipelineManagerTest ${NIFTK_DATA_DIR}/Input/volunteers/16856/16856-002-1.img)
