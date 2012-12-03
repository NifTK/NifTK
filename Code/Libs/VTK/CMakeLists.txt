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
#  Last Changed      : $LastChangedDate: 2010-07-28 12:37:56 +0100 (Wed, 28 Jul 2010) $ 
#  Revision          : $Revision: 8038 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

SET(niftkVTK_SRCS
  Common/vtkFunctions.cpp
)

IF(NIFTYREG_FOUND)
  ADD_DEFINITIONS(-DUSE_NIFTYREG)
  LINK_DIRECTORIES(${NIFTYREG_DIR}/lib)
  INCLUDE_DIRECTORIES(${NIFTYREG_DIR}/include)
      
  SET(niftkVTK_SRCS
    ${niftkVTK_SRCS}
    niftkF3DControlGridToVTKPolyData.cxx
    )

  SET(NIFTYREG_LIBRARIES
    _reg_KLdivergence
    _reg_blockMatching
    _reg_femTransformation
    _reg_globalTransformation
    _reg_localTransformation
    _reg_maths
    _reg_mutualinformation
    _reg_resampling
    _reg_ssd
    _reg_tools
    _reg_ReadWriteImage
    reg_png
    png
    reg_nrrd
    reg_NrrdIO
    )
ENDIF(NIFTYREG_FOUND)

ADD_LIBRARY(niftkVTK ${niftkVTK_SRCS})

IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    ADD_DEFINITIONS(-DNIFTKVTK_WINDOWS_EXPORT)
  ENDIF(WIN32)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(niftkVTK
  niftkcommon
  ${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
  ${Boost_LIBRARIES}  
  ${NIFTYREG_LIBRARIES}
) 

MITK_INSTALL(TARGETS niftkVTK)
