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
#  Last Changed      : $LastChangedDate: 2011-05-07 15:05:50 +0100 (Sat, 07 May 2011) $ 
#  Revision          : $Revision: 6093 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# We need to run the GUI generator.
# This lists the .ui files that need the pre-processor run on them

SET(niftkqt_GUI_UI
IntegerSpinBoxAndSliderWidget.ui
)
QT4_WRAP_UI(niftkqt_GUI_HEAD ${niftkqt_GUI_UI})

# We run moc to replace/generate any Q_OBJECT macros and the like.
# So this lists the files that need moc.

SET(niftkqt_MOC_HEADERS
IntegerSpinBoxAndSliderWidget.h
SliceSelectWidget.h
MagnificationFactorWidget.h
)

QT4_WRAP_CPP(niftkqt_MOC_SRCS ${niftkqt_MOC_HEADERS})
QT4_GENERATE_MOC(niftkqt_MOC_SRCS ${niftkqt_MOC_HEADERS})

# These are generic cpp files that don't need pre-processing.

SET(niftkqt_SRCS
WidgetHelper.cpp
IntegerSpinBoxAndSliderWidget.cpp
SliceSelectWidget.cpp
MagnificationFactorWidget.cpp
)

ADD_LIBRARY(niftkQt
  ${niftkqt_MOC_SRCS}
  ${niftkqt_GUI_HEAD}
  ${niftkqt_SRCS} 
)

IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    ADD_DEFINITIONS(-DNIFTKQT_WINDOWS_EXPORT)
  ENDIF(WIN32)
ENDIF(BUILD_SHARED_LIBS)
  
TARGET_LINK_LIBRARIES(niftkQt
  ${QT_LIBRARIES}  
) 


IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    INSTALL(TARGETS niftkQt  
      RUNTIME DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
      COMPONENT libraries 
    )
    INSTALL(TARGETS niftkQt  
      ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
      COMPONENT libraries 
    )    
  ELSE(WIN32)
    INSTALL(TARGETS niftkQt 
      LIBRARY DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
      ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
      COMPONENT libraries 
    )
  ENDIF(WIN32)
ELSE(BUILD_SHARED_LIBS)
  INSTALL(TARGETS niftkQt 
    LIBRARY DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
    ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
    COMPONENT libraries 
  )
ENDIF(BUILD_SHARED_LIBS)
