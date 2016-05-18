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

set(H_FILES
  Algorithms/niftkGeneralSegmentorCommands.h
  Algorithms/niftkGeneralSegmentorPipeline.txx
  Algorithms/niftkGeneralSegmentorUtils.h
  Algorithms/niftkGeneralSegmentorUtils.txx
  Algorithms/niftkMorphologicalSegmentorPipelineManager.h
  Interactions/niftkToolFactoryMacros.h
)

set(CPP_FILES
  DataManagement/niftkMIDASDataNodeNameStringFilter.cxx
  Algorithms/niftkMorphologicalSegmentorPipelineManager.cxx
  Algorithms/niftkGeneralSegmentorCommands.cxx
  Algorithms/niftkGeneralSegmentorPipeline.cxx
  Algorithms/niftkGeneralSegmentorPipelineCache.cxx
  Algorithms/niftkGeneralSegmentorUtils.cxx
  Interactions/niftkMIDASTool.cxx
  Interactions/niftkMIDASContourToolEventInterface.cxx
  Interactions/niftkMIDASContourToolOpAccumulateContour.cxx
  Interactions/niftkMIDASContourTool.cxx
  Interactions/niftkMIDASDrawToolEventInterface.cxx
  Interactions/niftkMIDASDrawToolOpEraseContour.cxx
  Interactions/niftkMIDASDrawTool.cxx
  Interactions/niftkMIDASEventFilter.cxx
  Interactions/niftkMIDASPointSetInteractor.cxx
  Interactions/niftkMIDASPointSetDataInteractor.cxx
  Interactions/niftkMIDASPolyToolEventInterface.cxx
  Interactions/niftkMIDASPolyToolOpAddToFeedbackContour.cxx
  Interactions/niftkMIDASPolyToolOpUpdateFeedbackContour.cxx
  Interactions/niftkMIDASPolyTool.cxx
  Interactions/niftkMIDASSeedTool.cxx
  Interactions/niftkMIDASPosnTool.cxx
  Interactions/niftkMIDASPaintbrushToolEventInterface.cxx
  Interactions/niftkMIDASPaintbrushToolOpEditImage.cxx
  Interactions/niftkMIDASPaintbrushTool.cxx
  Interactions/niftkMIDASRendererFilter.cxx
  Interactions/niftkMIDASStateMachine.cxx
  Interactions/niftkMIDASToolKeyPressStateMachine.cxx
)

set(RESOURCE_FILES
  Interactions/DisplayConfigMIDASTool.xml
  Interactions/DisplayConfigMIDASPaintbrushTool.xml
  Interactions/DnDDisplayConfigMIDASTool.xml
  Interactions/DnDDisplayConfigMIDASPaintbrushTool.xml
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
