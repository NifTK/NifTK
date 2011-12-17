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
 
SET(NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES 
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
  
ADD_EXECUTABLE(niftkUpdateParcellation niftkUpdateParcellation.cxx )
TARGET_LINK_LIBRARIES(niftkUpdateParcellation ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkUpdateParcellation RUNTIME DESTINATION bin COMPONENT applications)
  
ADD_EXECUTABLE(niftkLabelConnectivityToMatrix niftkLabelConnectivityToMatrix.cxx )
TARGET_LINK_LIBRARIES(niftkLabelConnectivityToMatrix ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkLabelConnectivityToMatrix RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkMergeConnectivityWithSpheres niftkMergeConnectivityWithSpheres.cxx )
TARGET_LINK_LIBRARIES(niftkMergeConnectivityWithSpheres ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkMergeConnectivityWithSpheres RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkShrinkLabelConnectivityFile niftkShrinkLabelConnectivityFile.cxx )
TARGET_LINK_LIBRARIES(niftkShrinkLabelConnectivityFile ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkShrinkLabelConnectivityFile RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkDropLinesFromPolyData niftkDropLinesFromPolyData.cxx )
TARGET_LINK_LIBRARIES(niftkDropLinesFromPolyData ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkDropLinesFromPolyData RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkSplitConnectedSpheres niftkSplitConnectedSpheres.cxx )
TARGET_LINK_LIBRARIES(niftkSplitConnectedSpheres ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkSplitConnectedSpheres RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkConvertFreeSurferThicknessToPolyData niftkConvertFreeSurferThicknessToPolyData.cxx )
TARGET_LINK_LIBRARIES(niftkConvertFreeSurferThicknessToPolyData ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkConvertFreeSurferThicknessToPolyData RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkExtractStatsFromParcellatedSurface niftkExtractStatsFromParcellatedSurface.cxx )
TARGET_LINK_LIBRARIES(niftkExtractStatsFromParcellatedSurface ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkExtractStatsFromParcellatedSurface RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkPasteDataToSurface niftkPasteDataToSurface.cxx )
TARGET_LINK_LIBRARIES(niftkPasteDataToSurface ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkPasteDataToSurface RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkCheckConnectivityFile niftkCheckConnectivityFile.cxx )
TARGET_LINK_LIBRARIES(niftkCheckConnectivityFile ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkCheckConnectivityFile RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkTransferClosestLabels niftkTransferClosestLabels.cxx )
TARGET_LINK_LIBRARIES(niftkTransferClosestLabels ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkTransferClosestLabels RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkTransferCorrespondingLabels niftkTransferCorrespondingLabels.cxx )
TARGET_LINK_LIBRARIES(niftkTransferCorrespondingLabels ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkTransferCorrespondingLabels RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkSubtractCorrespondingLabels niftkSubtractCorrespondingLabels.cxx )
TARGET_LINK_LIBRARIES(niftkSubtractCorrespondingLabels ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkSubtractCorrespondingLabels RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkEvaluateLabelHistogram niftkEvaluateLabelHistogram.cxx )
TARGET_LINK_LIBRARIES(niftkEvaluateLabelHistogram ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkEvaluateLabelHistogram RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkMoveBoundaryLabels niftkMoveBoundaryLabels.cxx )
TARGET_LINK_LIBRARIES(niftkMoveBoundaryLabels ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkMoveBoundaryLabels RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkLabelledSurfaceDiceScore niftkLabelledSurfaceDiceScore.cxx )
TARGET_LINK_LIBRARIES(niftkLabelledSurfaceDiceScore ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkLabelledSurfaceDiceScore RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkCompareLabelValues niftkCompareLabelValues.cxx )
TARGET_LINK_LIBRARIES(niftkCompareLabelValues ${NIFTK_PROTOTYPE_MJC_PARCELLATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkCompareLabelValues RUNTIME DESTINATION bin COMPONENT applications)
