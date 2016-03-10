/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMIDASPaintbrushTool.h"

#include <vtkImageData.h>

#include <mitkBaseRenderer.h>
#include <mitkDataStorageUtils.h>
#include <mitkDisplayInteractor.h>
#include <mitkImageAccessByItk.h>
#include <mitkInstantiateAccessFunctions.h>
#include <mitkITKImageImport.h>
#include <mitkITKRegionParametersDataNodeProperty.h>
#include <mitkPointUtils.h>
#include <mitkRenderingManager.h>
#include <mitkToolManager.h>
#include <mitkUndoController.h>
#include <mitkVector.h>

#include "niftkMIDASPaintbrushTool.xpm"
#include "niftkMIDASPaintbrushToolOpEditImage.h"
#include "niftkMIDASPaintbrushToolEventInterface.h"

// MicroServices
#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleRegistry.h>

const std::string niftk::MIDASPaintbrushTool::EROSIONS_ADDITIONS_NAME = "MIDAS_EDITS_EROSIONS_ADDITIONS";
const std::string niftk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS_NAME = "MIDAS_EDITS_EROSIONS_SUBTRACTIONS";
const std::string niftk::MIDASPaintbrushTool::DILATIONS_ADDITIONS_NAME = "MIDAS_EDITS_DILATIONS_ADDITIONS";
const std::string niftk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS_NAME = "MIDAS_EDITS_DILATIONS_SUBTRACTIONS";

const std::string niftk::MIDASPaintbrushTool::REGION_PROPERTY_NAME = std::string("midas.morph.editing.region");
const mitk::OperationType niftk::MIDASPaintbrushTool::MIDAS_PAINTBRUSH_TOOL_OP_EDIT_IMAGE = 320410;

namespace niftk
{
  MITK_TOOL_MACRO(NIFTKMIDAS_EXPORT, MIDASPaintbrushTool, "MIDAS Paintbrush Tool");
}

niftk::MIDASPaintbrushTool::MIDASPaintbrushTool()
: mitk::SegTool2D("")
, m_Interface(NULL)
, m_CursorSize(1)
, m_WorkingImageGeometry(NULL)
, m_WorkingImage(NULL)
, m_ErosionMode(true)
{
  m_Interface = niftk::MIDASPaintbrushToolEventInterface::New();
  m_Interface->SetMIDASPaintbrushTool( this );
}

niftk::MIDASPaintbrushTool::~MIDASPaintbrushTool()
{
}

void niftk::MIDASPaintbrushTool::InitializeStateMachine()
{
  try
  {
    this->LoadStateMachine("MIDASPaintbrushTool.xml", us::GetModuleContext()->GetModule());
    this->SetEventConfig("MIDASPaintbrushToolConfig.xml", us::GetModuleContext()->GetModule());
  }
  catch( const std::exception& e )
  {
    MITK_ERROR << "Could not load statemachine pattern MIDASPaintbrushTool.xml with exception: " << e.what();
  }
}

void niftk::MIDASPaintbrushTool::ConnectActionsAndFunctions()
{
  CONNECT_FUNCTION("startAddingAddition", StartAddingAddition);
  CONNECT_FUNCTION("keepAddingAddition", KeepAddingAddition);
  CONNECT_FUNCTION("stopAddingAddition", StopAddingAddition);
  CONNECT_FUNCTION("startAddingSubtraction", StartAddingSubtraction);
  CONNECT_FUNCTION("keepAddingSubtraction", KeepAddingSubtraction);
  CONNECT_FUNCTION("stopAddingSubtraction", StopAddingSubtraction);
  CONNECT_FUNCTION("startRemovingSubtraction", StartRemovingSubtraction);
  CONNECT_FUNCTION("keepRemovingSubtraction", KeepRemovingSubtraction);
  CONNECT_FUNCTION("stopRemovingSubtraction", StopRemovingSubtraction);
}

const char* niftk::MIDASPaintbrushTool::GetName() const
{
  return "Paintbrush";
}

const char** niftk::MIDASPaintbrushTool::GetXPM() const
{
  return niftkMIDASPaintbrushTool_xpm;
}

void niftk::MIDASPaintbrushTool::Activated()
{
  Superclass::Activated();

  CursorSizeChanged.Send(m_CursorSize);

  // As a legacy solution the display interaction of the new interaction framework is disabled here  to avoid conflicts with tools
  // Note: this only affects InteractionEventObservers (formerly known as Listeners) all DataNode specific interaction will still be enabled
  m_DisplayInteractorConfigs.clear();
  std::vector<us::ServiceReference<InteractionEventObserver> > listEventObserver = us::GetModuleContext()->GetServiceReferences<InteractionEventObserver>();
  for (std::vector<us::ServiceReference<InteractionEventObserver> >::iterator it = listEventObserver.begin(); it != listEventObserver.end(); ++it)
  {
    mitk::DisplayInteractor* displayInteractor = dynamic_cast<mitk::DisplayInteractor*>(
                                                    us::GetModuleContext()->GetService<InteractionEventObserver>(*it));
    if (displayInteractor)
    {
      if (std::strcmp(displayInteractor->GetNameOfClass(), "DnDDisplayInteractor") == 0)
      {
        // remember the original configuration
        m_DisplayInteractorConfigs.insert(std::make_pair(*it, displayInteractor->GetEventConfig()));
        // here the alternative configuration is loaded
        displayInteractor->SetEventConfig("DnDDisplayConfigMIDASPaintbrushTool.xml", us::GetModuleContext()->GetModule());
      }
    }
  }
}

void niftk::MIDASPaintbrushTool::Deactivated()
{
  // Re-enabling InteractionEventObservers that have been previously disabled for legacy handling of Tools
  // in new interaction framework
  for (std::map<us::ServiceReferenceU, mitk::EventConfig>::iterator it = m_DisplayInteractorConfigs.begin();
       it != m_DisplayInteractorConfigs.end(); ++it)
  {
    if (it->first)
    {
      mitk::DisplayInteractor* displayInteractor = static_cast<mitk::DisplayInteractor*>(
                                               us::GetModuleContext()->GetService<mitk::InteractionEventObserver>(it->first));
      if (displayInteractor)
      {
        if (std::strcmp(displayInteractor->GetNameOfClass(), "DnDDisplayInteractor") == 0)
        {
          // here the regular configuration is loaded again
          displayInteractor->SetEventConfig(it->second);
        }
      }
    }
  }
  m_DisplayInteractorConfigs.clear();

  mitk::Tool::Deactivated();
}

bool niftk::MIDASPaintbrushTool::FilterEvents(mitk::InteractionEvent* event, mitk::DataNode* dataNode)
{
  return this->CanHandleEvent(event);
}

int niftk::MIDASPaintbrushTool::GetCursorSize() const
{
  return m_CursorSize;
}

void niftk::MIDASPaintbrushTool::SetCursorSize(int cursorSize)
{
  m_CursorSize = cursorSize;
}

bool niftk::MIDASPaintbrushTool::GetErosionMode() const
{
  return m_ErosionMode;
}

void niftk::MIDASPaintbrushTool::SetErosionMode(bool erosionMode)
{
  m_ErosionMode = erosionMode;;
}

void niftk::MIDASPaintbrushTool::GetListOfAffectedVoxels(
    const mitk::PlaneGeometry& planeGeometry,
    mitk::Point3D& currentPoint,
    mitk::Point3D& previousPoint,
    ProcessorType &processor)
{
  assert(m_WorkingImage);
  assert(m_WorkingImageGeometry);

  processor.ClearList();

  // Need to work out which two axes we are working in, and bail out if it fails.
  int affectedDimension( -1 );
  int affectedSlice( -1 );

  if (!(SegTool2D::DetermineAffectedImageSlice(m_WorkingImage, &planeGeometry, affectedDimension, affectedSlice )))
  {
    return;
  }

  int whichTwoAxesInVoxelSpace[2];
  if (affectedDimension == 0)
  {
    whichTwoAxesInVoxelSpace[0] = 1;
    whichTwoAxesInVoxelSpace[1] = 2;
  }
  else if (affectedDimension == 1)
  {
    whichTwoAxesInVoxelSpace[0] = 0;
    whichTwoAxesInVoxelSpace[1] = 2;
  }
  else if (affectedDimension == 2)
  {
    whichTwoAxesInVoxelSpace[0] = 0;
    whichTwoAxesInVoxelSpace[1] = 1;
  }

  // Get size, for now using VTK spacing.
  mitk::Image::Pointer nonConstImage = const_cast<mitk::Image*>(m_WorkingImage);
  vtkImageData* vtkImage = nonConstImage->GetVtkImageData(0, 0);
  double *spacing = vtkImage->GetSpacing();

  // Work out the smallest dimension and hence the step size along the line
  double stepSize = mitk::CalculateStepSize(spacing);

  mitk::Point3D mostRecentPoint = previousPoint;
  mitk::Point3D vectorDifference;
  mitk::Point3D projectedPointIn3DVoxels;
  mitk::Point3D previousProjectedPointIn3DVoxels;
  mitk::Point3D cursorPointIn3DVoxels;
  itk::Index<3> affectedVoxel;

//  dont forget to set projectedPointIn3DVoxels equal invalid value, then track
//  all new points and only add to processor list if different to previous.

  mitk::GetDifference(currentPoint, mostRecentPoint, vectorDifference);
  double length = mitk::GetSquaredDistanceBetweenPoints(currentPoint, mostRecentPoint);

  // So, all remaining work is only done if we had a vector with some length to it.
  if (length > 0)
  {
    // Calculate how many steps we are taking along vector, and hence normalize
    // the vectorDifference to be a direction vector for each step.
    length = sqrt(length);
    int steps = (int)(length / stepSize);

    // All remaining work should be done only if we are going
    // to step along vector (otherwise infinite loop later).
    if (steps > 0)
    {
      previousProjectedPointIn3DVoxels[0] = std::numeric_limits<float>::max();
      previousProjectedPointIn3DVoxels[1] = std::numeric_limits<float>::max();
      previousProjectedPointIn3DVoxels[2] = std::numeric_limits<float>::max();

      // Normalise the vector difference to make it a direction vector for stepping along the line.
      for (int i = 0; i < 3; i++)
      {
        vectorDifference[i] /= length;
        vectorDifference[i] *= stepSize;
      }

      for (int k = 0; k < steps; k++)
      {
        for (int i = 0; i < 3; i++)
        {
          mostRecentPoint[i] += vectorDifference[i];
        }

        // Convert to voxels and round.
        m_WorkingImageGeometry->WorldToIndex( mostRecentPoint, projectedPointIn3DVoxels );
        for (int i = 0; i < 3; i++)
        {
          projectedPointIn3DVoxels[i] = (int)(projectedPointIn3DVoxels[i] + 0.5);
        }

        // We only add this point to the list if it is different to previous.
        if (projectedPointIn3DVoxels != previousProjectedPointIn3DVoxels)
        {
          // Check we are not outside image before adding any index.
          // This means if the stroke of the mouse, or the size of
          // the cross is outside of the image, we will not crash.
          if (m_WorkingImageGeometry->IsIndexInside(projectedPointIn3DVoxels))
          {
            for (int i = 0; i < 3; i++)
            {
              affectedVoxel[i] = (long int)projectedPointIn3DVoxels[i];
            }
            processor.AddToList(affectedVoxel);
          }

          int actualCursorSize = m_CursorSize - 1;
          if (actualCursorSize > 0)
          {
            for (int dimension = 0; dimension < 2; dimension++)
            {
              cursorPointIn3DVoxels = projectedPointIn3DVoxels;

              // Now draw a cross centred at projectedPointIn3DVoxels, but don't do centre, as it is done above.
              for (int offset = -actualCursorSize; offset <= actualCursorSize; offset++)
              {
                if (offset != 0)
                {
                  cursorPointIn3DVoxels[whichTwoAxesInVoxelSpace[dimension]] = projectedPointIn3DVoxels[whichTwoAxesInVoxelSpace[dimension]] + offset;

                  for (int i = 0; i < 3; i++)
                  {
                    affectedVoxel[i] = (long int)cursorPointIn3DVoxels[i];
                  }

                  // Check we are not outside image before adding any index.
                  // This means if the stroke of the mouse, or the size of
                  // the cross is outside of the image, we will not crash.
                  if (m_WorkingImageGeometry->IsIndexInside(affectedVoxel))
                  {
                    processor.AddToList(affectedVoxel);
                  }
                }
              }
            }
          }
          previousProjectedPointIn3DVoxels = projectedPointIn3DVoxels;
        } // end if projected point != previous projected point
      } // end for k, foreach step
    } // end if steps > 0
  } // end if length > 0
} // end function

bool niftk::MIDASPaintbrushTool::MarkInitialPosition(unsigned int dataIndex, mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  mitk::DataNode* workingNode = m_ToolManager->GetWorkingData(dataIndex);
  if (!workingNode)
  {
    return false;
  }

  // Store these for later, as dynamic casts are slow. HOWEVER, IT IS NOT THREAD SAFE.
  m_WorkingImage = dynamic_cast<mitk::Image*>(workingNode->GetData());
  m_WorkingImageGeometry = m_WorkingImage->GetGeometry();

  // Make sure we have a valid position event, otherwise no point continuing.
  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  if (!positionEvent)
  {
    return false;
  }

  // Set reference data, but we don't draw anything at this stage
  m_MostRecentPointInMillimetres = positionEvent->GetPositionInWorld();
  return true;
}


bool niftk::MIDASPaintbrushTool::DoMouseMoved(mitk::StateMachineAction* action, mitk::InteractionEvent* event,
    int dataIndex,
    unsigned char valueForRedo,
    unsigned char valueForUndo

    )
{
  if (m_WorkingImage == NULL || m_WorkingImageGeometry == NULL)
  {
    return false;
  }

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  if (!positionEvent)
  {
    return false;
  }

  const mitk::PlaneGeometry* planeGeometry = dynamic_cast<const mitk::PlaneGeometry*>(positionEvent->GetSender()->GetCurrentWorldPlaneGeometry());
  if ( !planeGeometry )
  {
    return false;
  }

  mitk::DataNode* workingNode = m_ToolManager->GetWorkingData(dataIndex);
  assert(workingNode);

  mitk::Image::Pointer imageToWriteTo = static_cast<mitk::Image*>(workingNode->GetData());
  assert(imageToWriteTo);

  mitk::Point3D currentPoint = positionEvent->GetPositionInWorld();

  ProcessorType::Pointer processor = ProcessorType::New();
  this->GetListOfAffectedVoxels(*planeGeometry, currentPoint, m_MostRecentPointInMillimetres, (*processor));

  if (processor->GetNumberOfVoxels() > 0)
  {
    try
    {
      std::vector<int> boundingBox = processor->ComputeMinimalBoundingBox();
      this->SetValidRegion(dataIndex, boundingBox);

      MIDASPaintbrushToolOpEditImage *doOp = new MIDASPaintbrushToolOpEditImage(MIDAS_PAINTBRUSH_TOOL_OP_EDIT_IMAGE, true, dataIndex, valueForRedo, imageToWriteTo, workingNode, processor);
      MIDASPaintbrushToolOpEditImage *undoOp = new MIDASPaintbrushToolOpEditImage(MIDAS_PAINTBRUSH_TOOL_OP_EDIT_IMAGE, false, dataIndex, valueForUndo, imageToWriteTo, workingNode, processor);
      mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Edit Image");
      mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );

      this->ExecuteOperation(doOp);
    }
    catch( itk::ExceptionObject & err )
    {
      MITK_ERROR << "Failed to perform edit in niftkMIDASPaintrushTool due to:" << err << std::endl;
    }
  }

  positionEvent->GetSender()->GetRenderingManager()->RequestUpdateAll();

  m_MostRecentPointInMillimetres = currentPoint;
  return true;
}

int niftk::MIDASPaintbrushTool::GetDataIndex(bool isLeftMouseButton)
{
  int dataIndex = -1;

  if (isLeftMouseButton)
  {
    if (m_ErosionMode)
    {
      dataIndex = EROSIONS_ADDITIONS;
    }
    else
    {
      dataIndex = DILATIONS_ADDITIONS;
    }
  }
  else
  {
    if (m_ErosionMode)
    {
      dataIndex = EROSIONS_SUBTRACTIONS;
    }
    else
    {
      dataIndex = DILATIONS_SUBTRACTIONS;
    }
  }

  assert(dataIndex >= 0 && dataIndex <=3);
  return dataIndex;
}

bool niftk::MIDASPaintbrushTool::StartAddingAddition(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  int dataIndex = this->GetDataIndex(true);
  bool result = this->MarkInitialPosition(dataIndex, action, event);
  return result;
}

bool niftk::MIDASPaintbrushTool::KeepAddingAddition(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  int dataIndex = this->GetDataIndex(true);
  this->DoMouseMoved(action, event, dataIndex, 1, 0);
  return true;
}

bool niftk::MIDASPaintbrushTool::StopAddingAddition(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  int dataIndex = this->GetDataIndex(true);
  this->SetInvalidRegion(dataIndex);
  // The data is not actually modified here. We fire this event so that the pipeline is
  // updated once again after the interaction has finished. This is needed because during
  // the interaction the pipeline updates only the current slice, not the whole image,
  // because that would be too slow. When the interaction is finished (mouse button released)
  // we rerun last step of the pipeline. The pipeline decides whether it should update the
  // entire image or just the current slice, based on if there is a valid region set. (See
  // the SetInvalidRegion() call above.)
  this->SegmentationEdited.Send(dataIndex);
  return true;
}

bool niftk::MIDASPaintbrushTool::StartAddingSubtraction(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  int dataIndex = this->GetDataIndex(false);
  return this->MarkInitialPosition(dataIndex, action, event);
}

bool niftk::MIDASPaintbrushTool::KeepAddingSubtraction(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  int dataIndex = this->GetDataIndex(false);
  this->DoMouseMoved(action, event, dataIndex, 1, 0);
  return true;
}

bool niftk::MIDASPaintbrushTool::StopAddingSubtraction(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  int dataIndex = this->GetDataIndex(false);
  this->SetInvalidRegion(dataIndex);
  this->SegmentationEdited.Send(dataIndex);
  return true;
}

bool niftk::MIDASPaintbrushTool::StartRemovingSubtraction(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  int dataIndex = this->GetDataIndex(false);
  return this->MarkInitialPosition(dataIndex, action, event);
}

bool niftk::MIDASPaintbrushTool::KeepRemovingSubtraction(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  int dataIndex = this->GetDataIndex(false);
  this->DoMouseMoved(action, event, dataIndex, 0, 1);
  return true;
}

bool niftk::MIDASPaintbrushTool::StopRemovingSubtraction(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  int dataIndex = this->GetDataIndex(false);
  this->SetInvalidRegion(dataIndex);
  this->SegmentationEdited.Send(dataIndex);
  return true;
}

void niftk::MIDASPaintbrushTool::SetRegion(unsigned int dataIndex, bool valid, const std::vector<int>& boundingBox)
{
  mitk::DataNode* workingNode = m_ToolManager->GetWorkingData(dataIndex);
  assert(workingNode);

  // This property should always exist, as we create it when the volume is created.
  mitk::BaseProperty* baseProperty = workingNode->GetProperty(REGION_PROPERTY_NAME.c_str());
  mitk::ITKRegionParametersDataNodeProperty::Pointer prop = dynamic_cast<mitk::ITKRegionParametersDataNodeProperty*>(baseProperty);

  if (valid)
  {
    prop->SetITKRegionParameters(boundingBox);
    prop->SetValid(true);
  }
  else
  {
    // Put some fake volume in there. Doesn't matter what the volume is, as it is marked as Invalid anyway.
    prop->SetSize(1, 1, 1);
    prop->SetValid(false);
  }
}

void niftk::MIDASPaintbrushTool::SetInvalidRegion(unsigned int dataIndex)
{
  this->SetRegion(dataIndex, false);
}

void niftk::MIDASPaintbrushTool::SetValidRegion(unsigned int dataIndex, const std::vector<int>& boundingBox)
{
  this->SetRegion(dataIndex, true, boundingBox);
}

void niftk::MIDASPaintbrushTool::ExecuteOperation(mitk::Operation* operation)
{
  if (!operation) return;

  switch (operation->GetOperationType())
  {
  case MIDAS_PAINTBRUSH_TOOL_OP_EDIT_IMAGE:
    {
      MIDASPaintbrushToolOpEditImage *op = static_cast<MIDASPaintbrushToolOpEditImage*>(operation);
      unsigned char valueToWrite = op->GetValueToWrite();
      ProcessorType::Pointer processor = op->GetProcessor();
      mitk::Image::Pointer imageToEdit = op->GetImageToEdit();
      bool redo = op->IsRedo();

      typedef mitk::ImageToItk< ImageType > ImageToItkType;
      ImageToItkType::Pointer imageToEditToItk = ImageToItkType::New();
      imageToEditToItk->SetInput(imageToEdit);
      imageToEditToItk->Update();

      this->RunITKProcessor<mitk::Tool::DefaultSegmentationDataType, 3>(imageToEditToItk->GetOutput(), processor, redo, valueToWrite);

      imageToEditToItk = 0;
      imageToEdit = NULL;

      this->SegmentationEdited.Send(op->GetImageNumber());

      if (m_LastEventSender)
      {
        m_LastEventSender->GetRenderingManager()->RequestUpdateAll();
      }
      break;
    }
  default:;
  }
}

template<typename TPixel, unsigned int VImageDimension>
void niftk::MIDASPaintbrushTool::RunITKProcessor(
    itk::Image<TPixel, VImageDimension>* itkImage,
    ProcessorType::Pointer processor,
    bool redo,
    unsigned char valueToWrite
    )
{
  processor->SetDestinationImage(itkImage);
  processor->SetValue(valueToWrite);

  if (redo)
  {
    processor->Redo();
  }
  else
  {
    processor->Undo();
  }

  processor->SetDestinationImage(NULL);
}
