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

MITK_CREATE_MODULE(niftkMIDASGui
  INCLUDE_DIRS Events Dialogs LookupTables
  DEPENDS MitkCore MitkQtWidgets MitkSegmentationUI niftkCore niftkCoreGui niftkMIDAS
)

target_link_libraries(niftkMIDASGui PRIVATE niftkITK)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkMIDASGui)
endif()
