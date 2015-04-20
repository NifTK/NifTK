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

set(CPP_FILES
  Rendering/VLQt4Widget.cpp
  Rendering/OclTriangleSorter.cxx
  VLEditor/QmitkIGIVLEditor.cxx
)

set(MOC_H_FILES
  Rendering/VLQt4Widget.h
  VLEditor/QmitkIGIVLEditor.h
)

set(UI_FILES
  VLEditor/QmitkIGIVLEditor.ui
)

set(QRC_FILES
  Resources/niftkVL.qrc
)

if(CUDA_FOUND AND NIFTK_USE_CUDA)
  set(CPP_FILES
    ${CPP_FILES}
    Rendering/VLFramebufferToCUDA.cxx
  )
endif()
