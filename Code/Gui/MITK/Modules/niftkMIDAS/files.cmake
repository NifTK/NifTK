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
  Algorithms/mitkMIDASMorphologicalSegmentorPipelineManager.cxx
  DataManagement/mitkMIDASDataNodeNameStringFilter.cxx
  DataManagement/mitkMIDASNodeAddedVisibilitySetter.cxx
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
  Interactions/mitkMIDASDisplayInteractor.cxx
)

set(RESOURCE_FILES
  Interactions/DisplayConfigMIDASTool.xml
  Interactions/DisplayConfigMIDASPaintbrushTool.xml
)
