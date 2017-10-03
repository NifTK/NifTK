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
  Interactions/niftkToolKeyPressResponder.h
)

set(CPP_FILES
  DataManagement/niftkToolWorkingDataNameFilter.cxx
  Algorithms/niftkMorphologicalSegmentorPipelineManager.cxx
  Algorithms/niftkGeneralSegmentorCommands.cxx
  Algorithms/niftkGeneralSegmentorPipeline.cxx
  Algorithms/niftkGeneralSegmentorPipelineCache.cxx
  Algorithms/niftkGeneralSegmentorUtils.cxx
  Interactions/niftkContourTool.cxx
  Interactions/niftkContourToolEventInterface.cxx
  Interactions/niftkContourToolOpAccumulateContour.cxx
  Interactions/niftkDrawTool.cxx
  Interactions/niftkDrawToolEventInterface.cxx
  Interactions/niftkDrawToolOpEraseContour.cxx
  Interactions/niftkFilteringStateMachine.cxx
  Interactions/niftkPaintbrushTool.cxx
  Interactions/niftkPaintbrushToolEventInterface.cxx
  Interactions/niftkPaintbrushToolOpEditImage.cxx
  Interactions/niftkPointSetDataInteractor.cxx
  Interactions/niftkPolyTool.cxx
  Interactions/niftkPolyToolEventInterface.cxx
  Interactions/niftkPolyToolOpAddToFeedbackContour.cxx
  Interactions/niftkPolyToolOpUpdateFeedbackContour.cxx
  Interactions/niftkPosnTool.cxx
  Interactions/niftkSeedTool.cxx
  Interactions/niftkStateMachineEventFilter.cxx
  Interactions/niftkStateMachineRendererFilter.cxx
  Interactions/niftkTool.cxx
  Interactions/niftkToolKeyPressStateMachine.cxx
)

set(RESOURCE_FILES
  Interactions/mitkDisplayConfig_niftkTool.xml
  Interactions/mitkDisplayConfig_niftkPaintbrushTool.xml
  Interactions/niftkDnDDisplayConfig_niftkTool.xml
  Interactions/niftkDnDDisplayConfig_niftkPaintbrushTool.xml
  Interactions/niftkDrawTool.xml
  Interactions/niftkDrawToolConfig.xml
  Interactions/niftkPaintbrushTool.xml
  Interactions/niftkPaintbrushToolConfig.xml
  Interactions/niftkPointSetDataInteractor.xml
  Interactions/niftkPointSetDataInteractorConfig.xml
  Interactions/niftkPolyTool.xml
  Interactions/niftkPolyToolConfig.xml
  Interactions/niftkSeedTool.xml
  Interactions/niftkToolKeyPressStateMachine.xml
  Interactions/niftkToolKeyPressStateMachineConfig.xml
  Interactions/niftkToolPointSetDataInteractor.xml
)
