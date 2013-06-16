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

MITK_CREATE_MODULE_TESTS(LABELS niftkIGI)

mitkAddCustomModuleTest(SLS-mitkSurfaceBasedRegistrationTest mitkSurfaceBasedRegistrationTest ${NIFTK_DATA_DIR}/Input/maskedliver_points.mps ${NIFTK_DATA_DIR}/Input/liver.vtk)


