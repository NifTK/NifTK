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

MITK_CREATE_MODULE_TESTS(LABELS niftkDnDDisplay)

# Trac 1683 - When nodes created via segmentation tools, the data sets much be visible in viewers
#             such as the thumbnail viewer, and the segmentation views.
mitkAddCustomModuleTest(MIDAS-mitkSegmentationNodeAddedVisibilityTest mitkMIDASSegmentationNodeAddedVisibilityTest ${NIFTK_DATA_DIR}/Input/nv-11x11x11.nii)
