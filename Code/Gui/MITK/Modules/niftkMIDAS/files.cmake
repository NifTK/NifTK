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
  Algorithms/mitkNifTKCoreObjectFactory.cxx
  Algorithms/mitkMIDASMorphologicalSegmentorPipelineManager.cxx
  Common/mitkMIDASImageUtils.cxx
  Common/mitkMIDASOrientationUtils.cxx
  Common/mitkPointUtils.cxx
  DataManagement/mitkDataNodeBoolPropertyFilter.cxx
  DataManagement/mitkDataNodeStringPropertyFilter.cxx
  DataManagement/mitkDataStorageUtils.cxx
  DataManagement/mitkDataStorageListener.cxx
  DataManagement/mitkDataStoragePropertyListener.cxx
  DataManagement/mitkDataStorageVisibilityTracker.cxx
  DataManagement/mitkMIDASNodeAddedVisibilitySetter.cxx
  DataManagement/mitkMIDASDataNodeNameStringFilter.cxx
  DataManagement/mitkCoordinateAxesData.cxx
  Rendering/mitkCoordinateAxesVtkMapper3D.cxx
  DataNodeProperties/mitkAffineTransformParametersDataNodeProperty.cxx
  DataNodeProperties/mitkAffineTransformDataNodeProperty.cxx
  DataNodeProperties/mitkITKRegionParametersDataNodeProperty.cxx
  DataNodeProperties/mitkNamedLookupTableProperty.cxx
  Interactions/mitkMIDASTool.cxx
  Interactions/mitkMIDASContourToolEventInterface.cxx
  Interactions/mitkMIDASContourToolOpAccumulateContour.cxx
  Interactions/mitkMIDASContourTool.cxx
  Interactions/mitkMIDASDrawToolEventInterface.cxx
  Interactions/mitkMIDASDrawToolOpEraseContour.cxx
  Interactions/mitkMIDASDrawTool.cxx
  Interactions/mitkMIDASPointSetInteractor.cxx
  Interactions/mitkMIDASPolyToolEventInterface.cxx
  Interactions/mitkMIDASPolyToolOpAddToFeedbackContour.cxx
  Interactions/mitkMIDASPolyToolOpUpdateFeedbackContour.cxx
  Interactions/mitkMIDASPolyTool.cxx
  Interactions/mitkMIDASSeedTool.cxx
  Interactions/mitkMIDASPosnTool.cxx
  Interactions/mitkMIDASPaintbrushToolEventInterface.cxx
  Interactions/mitkMIDASPaintbrushToolOpEditImage.cxx
  Interactions/mitkMIDASPaintbrushTool.cxx
  Interactions/mitkMIDASViewKeyPressStateMachine.cxx
  Interactions/mitkMIDASToolKeyPressStateMachine.cxx
  IO/itkAnalyzeImageIO3160.cxx
  IO/itkDRCAnalyzeImageIO3160.cxx
  IO/itkNiftiImageIO3201.cxx
  IO/mitkNifTKItkImageFileReader.cxx
  IO/mitkNifTKItkImageFileIOFactory.cxx
  IO/itkPNMImageIOFactory.cxx
  IO/itkPNMImageIO.cxx
)
