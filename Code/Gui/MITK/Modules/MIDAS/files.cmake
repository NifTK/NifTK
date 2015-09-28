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
  DataManagement/mitkMIDASDataNodeNameStringFilter.cxx
  Algorithms/mitkMIDASMorphologicalSegmentorPipelineManager.cxx
  Interactions/mitkMIDASTool.cxx
  Interactions/mitkMIDASContourToolEventInterface.cxx
  Interactions/mitkMIDASContourToolOpAccumulateContour.cxx
  Interactions/mitkMIDASContourTool.cxx
  Interactions/mitkMIDASDrawToolEventInterface.cxx
  Interactions/mitkMIDASDrawToolOpEraseContour.cxx
  Interactions/mitkMIDASDrawTool.cxx
  Interactions/mitkMIDASEventFilter.cxx
  Interactions/mitkMIDASPointSetInteractor.cxx
  Interactions/mitkMIDASPointSetDataInteractor.cxx
  Interactions/mitkMIDASPolyToolEventInterface.cxx
  Interactions/mitkMIDASPolyToolOpAddToFeedbackContour.cxx
  Interactions/mitkMIDASPolyToolOpUpdateFeedbackContour.cxx
  Interactions/mitkMIDASPolyTool.cxx
  Interactions/mitkMIDASSeedTool.cxx
  Interactions/mitkMIDASPosnTool.cxx
  Interactions/mitkMIDASPaintbrushToolEventInterface.cxx
  Interactions/mitkMIDASPaintbrushToolOpEditImage.cxx
  Interactions/mitkMIDASPaintbrushTool.cxx
  Interactions/mitkMIDASRendererFilter.cxx
  Interactions/mitkMIDASStateMachine.cxx
  Interactions/mitkMIDASToolKeyPressStateMachine.cxx
)

set(RESOURCE_FILES
  Interactions/DisplayConfigMIDASTool.xml
  Interactions/DisplayConfigMIDASPaintbrushTool.xml
  Interactions/MIDASDrawTool.xml
  Interactions/MIDASDrawToolConfig.xml
  Interactions/MIDASPolyTool.xml
  Interactions/MIDASPolyToolConfig.xml
  Interactions/MIDASPaintbrushTool.xml
  Interactions/MIDASPaintbrushToolConfig.xml
  Interactions/MIDASToolPointSetInteractor.xml
  Interactions/MIDASToolPointSetDataInteractor.xml
  Interactions/MIDASToolPointSetDataInteractorConfig.xml
  Interactions/MIDASSeedTool.xml
  Interactions/MIDASSeedToolPointSetInteractor.xml
  Interactions/MIDASSeedToolPointSetDataInteractor.xml
  Interactions/MIDASSeedToolPointSetDataInteractorConfig.xml
  Interactions/MIDASToolKeyPressStateMachine.xml
  Interactions/MIDASToolKeyPressStateMachineConfig.xml
)
