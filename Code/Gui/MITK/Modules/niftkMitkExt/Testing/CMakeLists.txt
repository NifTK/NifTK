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
#  Last Changed      : $LastChangedDate$ 
#  Revision          : $Revision$
#  Last modified by  : $Author$
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

MITK_CREATE_MODULE_TESTS(LABELS niftkMitkExt)

# Test "As Acquired" with known images, default Coronal, which is not needed here, as we should be able to work it out.
mitkAddCustomModuleTest(mitkMIDASAsAcquiredOrientationTest_16856   mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/16856/16856-002-1.img 2 2)
mitkAddCustomModuleTest(mitkMIDASAsAcquiredOrientationTest_01719-a mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-a.img 2 0)
mitkAddCustomModuleTest(mitkMIDASAsAcquiredOrientationTest_01719-s mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-s.img 2 1)
mitkAddCustomModuleTest(mitkMIDASAsAcquiredOrientationTest_01719-c mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/01719/01719-012-c.img 2 2)

# Test "As Acquired", changing default to make sure we are not needing it, output should always be sagittal.
mitkAddCustomModuleTest(mitkMIDASAsAcquiredOrientationTest_matt-0    mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_nifti/mdeft.1.nii 0 1)
mitkAddCustomModuleTest(mitkMIDASAsAcquiredOrientationTest_matt-1    mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_nifti/mdeft.1.nii 1 1)
mitkAddCustomModuleTest(mitkMIDASAsAcquiredOrientationTest_matt-2    mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_nifti/mdeft.1.nii 2 1)
mitkAddCustomModuleTest(mitkMIDASAsAcquiredOrientationTest_matt-14    mitkMIDASAsAcquiredOrientationTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_nifti/mdeft.1.nii 14 1)


