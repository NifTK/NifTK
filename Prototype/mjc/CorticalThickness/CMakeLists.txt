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
#  Last Changed      : $LastChangedDate: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $ 
#  Revision          : $Revision: 7333 $
#  Last modified by  : $Author: ad $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/
 
SET(NIFTK_PROTOTYPE_MJC_CORTICAL_LINK_LIBRARIES 
  niftkcommon  
  niftkITK
  ITKCommon 
  ITKIO 
  ITKBasicFilters
  ITKAlgorithms
  ITKQuadEdgeMesh
  itkzlib
  vtkGraphics
  vtkCommon
  vtkIO
  ${Boost_LIBRARIES} 
  )
  
ADD_EXECUTABLE(niftkMapPolyDataLabelsToDataValues niftkMapPolyDataLabelsToDataValues.cxx )
TARGET_LINK_LIBRARIES(niftkMapPolyDataLabelsToDataValues ${NIFTK_PROTOTYPE_MJC_CORTICAL_LINK_LIBRARIES} )
INSTALL(TARGETS niftkMapPolyDataLabelsToDataValues RUNTIME DESTINATION bin COMPONENT applications)
