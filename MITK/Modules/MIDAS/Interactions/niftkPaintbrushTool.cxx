/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPaintbrushTool.h"

#include <vtkImageData.h>

#include <mitkBaseRenderer.h>
#include <mitkDisplayInteractor.h>
#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>
#include <mitkImageAccessByItk.h>
#include <mitkInstantiateAccessFunctions.h>
#include <mitkITKImageImport.h>
#include <mitkRenderingManager.h>
#include <mitkToolManager.h>
#include <mitkUndoController.h>
#include <mitkVector.h>

// MicroServices
#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleRegistry.h>

#include <niftkDataStorageUtils.h>
#include <niftkInteractionEventObserverMutex.h>
#include <niftkITKRegionParametersDataNodeProperty.h>
#include <niftkPointUtils.h>

#include "niftkPaintbrushTool.xpm"
#include "niftkPaintbrushToolOpEditImage.h"
#include "niftkPaintbrushToolEventInterface.h"
#include "niftkToolFactoryMacros.h"

NIFTK_TOOL_MACRO(NIFTKMIDAS_EXPORT, PaintbrushTool, "Paintbrush Tool")

namespace niftk
{

const std::string PaintbrushTool::EROSIONS_ADDITIONS_NAME = "MIDAS_EDITS_EROSIONS_ADDITIONS";
const std::string PaintbrushTool::EROSIONS_SUBTRACTIONS_NAME = "MIDAS_EDITS_EROSIONS_SUBTRACTIONS";
const std::string PaintbrushTool::DILATIONS_ADDITIONS_NAME = "MIDAS_EDITS_DILATIONS_ADDITIONS";
const std::string PaintbrushTool::DILATIONS_SUBTRACTIONS_NAME = "MIDAS_EDITS_DILATIONS_SUBTRACTIONS";

const std::string PaintbrushTool::REGION_PROPERTY_NAME = std::string("midas.morph.editing.region");
const mitk::OperationType PaintbrushTool::MIDAS_PAINTBRUSH_TOOL_OP_EDIT_IMAGE = 320410;


//-----------------------------------------------------------------------------
PaintbrushTool::PaintbrushTool()
  : mitk::SegTool2D(""),
    m_Interface(nullptr),
    m_EraserSize(1),
    m_EraserVisible(false),
    m_WorkingImageGeometry(nullptr),
    m_WorkingImage(nullptr),
    m_ErosionMode(true),
    m_AddingAdditionInProgress(false),
    m_AddingSubtractionInProgress(false),
    m_RemovingSubtractionInProgress(false)
{
  m_Interface = PaintbrushToolEventInterface::New();
  m_Interface->SetPaintbrushTool(this);

  m_EraserPosition.Fill(0.0);

  m_EraserCursor = mitk::PlanarEllipse::New();
  m_EraserCursor->SetTreatAsCircle(false);
  mitk::Point2D position;
  position[0] = -0.5;
  position[1] = 0.0;
  m_EraserCursor->PlaceFigure(position);
  position[0] = 0.5;
  m_EraserCursor->SetCurrentControlPoint(position);
  position[0] = 0.0;
  position[1] = -0.5;
  m_EraserCursor->AddControlPoint(position);
  position[1] = 0.5;
  m_EraserCursor->AddControlPoint(position);
  this->SetEraserSize(m_EraserSize);

  m_EraserCursorNode = mitk::DataNode::New();
  m_EraserCursorNode->SetData(m_EraserCursor);
  m_EraserCursorNode->SetName("Paintbrush tool eraser");
  m_EraserCursorNode->SetBoolProperty("helper object", true);
  m_EraserCursorNode->SetBoolProperty("includeInBoundingBox", false);
  m_EraserCursorNode->SetBoolProperty("planarfigure.drawcontrolpoints", false);
  m_EraserCursorNode->SetBoolProperty("planarfigure.drawname", false);
  m_EraserCursorNode->SetBoolProperty("planarfigure.drawoutline", false);
  m_EraserCursorNode->SetBoolProperty("planarfigure.drawshadow", false);
  mitk::Color eraserColor;
  eraserColor.Set(1.0f, static_cast<float>(165.0 / 255.0), 0.0f); // orange like the segmentation below.
  m_EraserCursorNode->SetColor(eraserColor);
}


//-----------------------------------------------------------------------------
PaintbrushTool::~PaintbrushTool()
{
}


//-----------------------------------------------------------------------------
void PaintbrushTool::InitializeStateMachine()
{
  try
  {
    this->LoadStateMachine("niftkPaintbrushTool.xml", us::GetModuleContext()->GetModule());
    this->SetEventConfig("niftkPaintbrushToolConfig.xml", us::GetModuleContext()->GetModule());
  }
  catch( const std::exception& e )
  {
    MITK_ERROR << "Could not load statemachine pattern niftkPaintbrushTool.xml with exception: " << e.what();
  }
}


//-----------------------------------------------------------------------------
void PaintbrushTool::ConnectActionsAndFunctions()
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


//-----------------------------------------------------------------------------
const char* PaintbrushTool::GetName() const
{
  return "Paintbrush";
}


//-----------------------------------------------------------------------------
const char** PaintbrushTool::GetXPM() const
{
  return niftkPaintbrushTool_xpm;
}


//-----------------------------------------------------------------------------
void PaintbrushTool::Activated()
{
  Superclass::Activated();

  EraserSizeChanged.Send(m_EraserSize);

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
        displayInteractor->SetEventConfig("niftkDnDDisplayConfig_niftkPaintbrushTool.xml", us::GetModuleContext()->GetModule());
      }
    }
  }
}


//-----------------------------------------------------------------------------
void PaintbrushTool::Deactivated()
{
  if (m_AddingAdditionInProgress)
  {
    this->StopAddingAddition(nullptr, nullptr);
    this->ResetToStartState();
  }
  if (m_AddingSubtractionInProgress)
  {
    this->StopAddingSubtraction(nullptr, nullptr);
    this->ResetToStartState();
  }
  if (m_RemovingSubtractionInProgress)
  {
    this->StopRemovingSubtraction(nullptr, nullptr);
    this->ResetToStartState();
  }

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


//-----------------------------------------------------------------------------
bool PaintbrushTool::FilterEvents(mitk::InteractionEvent* event, mitk::DataNode* dataNode)
{
  return this->CanHandleEvent(event);
}


//-----------------------------------------------------------------------------
mitk::Point2D PaintbrushTool::GetEraserPosition() const
{
  return m_EraserPosition;
}


//-----------------------------------------------------------------------------
void PaintbrushTool::SetEraserPosition(const mitk::Point2D& eraserPosition)
{
  if (eraserPosition != m_EraserPosition)
  {
    m_EraserPosition = eraserPosition;

    this->UpdateEraserCursor();
  }
}


//-----------------------------------------------------------------------------
int PaintbrushTool::GetEraserSize() const
{
  return m_EraserSize;
}


//-----------------------------------------------------------------------------
void PaintbrushTool::SetEraserSize(int eraserSize)
{
  if (eraserSize != m_EraserSize)
  {
    m_EraserSize = eraserSize;

    this->UpdateEraserCursor();
  }
}


//-----------------------------------------------------------------------------
void PaintbrushTool::UpdateEraserCursor()
{
  mitk::BaseRenderer* renderer = mitk::GlobalInteraction::GetInstance()->GetFocusManager()->GetFocused();
  mitk::Point2D voxelSize;
  voxelSize.Fill(1.0);
  renderer->GetCurrentWorldPlaneGeometry()->IndexToWorld(voxelSize, voxelSize);

  mitk::Point2D position = m_EraserPosition;
  m_EraserCursor->SetControlPoint(0, position);

  double radius = m_EraserSize / 2.0;
  position[0] = m_EraserPosition[0] + radius * voxelSize[0];
  m_EraserCursor->SetControlPoint(1, position);

  position[0] = m_EraserPosition[0];
  position[1] = m_EraserPosition[1] + radius * voxelSize[1];
  m_EraserCursor->SetControlPoint(2, position);
}


//-----------------------------------------------------------------------------
bool PaintbrushTool::GetErosionMode() const
{
  return m_ErosionMode;
}


//-----------------------------------------------------------------------------
void PaintbrushTool::SetErosionMode(bool erosionMode)
{
  m_ErosionMode = erosionMode;;
}


//-----------------------------------------------------------------------------
void PaintbrushTool::GetListOfAffectedVoxels(
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
  double stepSize = niftk::CalculateStepSize(spacing);

  mitk::Point3D mostRecentPoint = previousPoint;
  mitk::Point3D vectorDifference;
  mitk::Point3D projectedPointIn3DVoxels;
  mitk::Point3D previousProjectedPointIn3DVoxels;
  mitk::Point3D cursorPointIn3DVoxels;
  itk::Index<3> affectedVoxel;

//  dont forget to set projectedPointIn3DVoxels equal invalid value, then track
//  all new points and only add to processor list if different to previous.

  niftk::GetDifference(currentPoint, mostRecentPoint, vectorDifference);
  double length = niftk::GetSquaredDistanceBetweenPoints(currentPoint, mostRecentPoint);

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

          int eraserRadius = m_EraserSize / 2;
          if (eraserRadius > 0)
          {
            for (int dimension = 0; dimension < 2; dimension++)
            {
              cursorPointIn3DVoxels = projectedPointIn3DVoxels;

              // Now draw a cross centred at projectedPointIn3DVoxels, but don't do centre, as it is done above.
              for (int offset = -eraserRadius; offset <= eraserRadius; ++offset)
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


//-----------------------------------------------------------------------------
bool PaintbrushTool::MarkInitialPosition(unsigned int dataIndex, mitk::StateMachineAction* action, mitk::InteractionEvent* event)
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


//-----------------------------------------------------------------------------
bool PaintbrushTool::DoMouseMoved(mitk::StateMachineAction* action, mitk::InteractionEvent* event,
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

      PaintbrushToolOpEditImage *doOp = new PaintbrushToolOpEditImage(MIDAS_PAINTBRUSH_TOOL_OP_EDIT_IMAGE, true, dataIndex, valueForRedo, imageToWriteTo, workingNode, processor);
      PaintbrushToolOpEditImage *undoOp = new PaintbrushToolOpEditImage(MIDAS_PAINTBRUSH_TOOL_OP_EDIT_IMAGE, false, dataIndex, valueForUndo, imageToWriteTo, workingNode, processor);
      mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Edit Image");
      mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );

      this->ExecuteOperation(doOp);
    }
    catch( itk::ExceptionObject & err )
    {
      MITK_ERROR << "Failed to perform edit in niftkPaintrushTool due to:" << err << std::endl;
    }
  }

  positionEvent->GetSender()->GetRenderingManager()->RequestUpdateAll();

  m_MostRecentPointInMillimetres = currentPoint;
  return true;
}


//-----------------------------------------------------------------------------
int PaintbrushTool::GetDataIndex(bool isLeftMouseButton)
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

  assert(dataIndex != -1);
  return dataIndex;
}


//-----------------------------------------------------------------------------
bool PaintbrushTool::StartAddingAddition(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  m_AddingAdditionInProgress = true;
  InteractionEventObserverMutex::GetInstance()->Lock(this);

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  assert(positionEvent);

  mitk::BaseRenderer* renderer = positionEvent->GetSender();
  const mitk::PlaneGeometry* planeGeometry = renderer->GetCurrentWorldPlaneGeometry();
  m_EraserCursor->SetPlaneGeometry(const_cast<mitk::PlaneGeometry*>(planeGeometry));
  mitk::Point2D position;
  planeGeometry->Map(positionEvent->GetPositionInWorld(), position);

  this->SetEraserPosition(position);
  this->SetEraserVisible(true, renderer);
  renderer->RequestUpdate();

  int dataIndex = this->GetDataIndex(true);
  bool result = this->MarkInitialPosition(dataIndex, action, event);
  return result;
}


//-----------------------------------------------------------------------------
bool PaintbrushTool::KeepAddingAddition(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  assert(m_AddingAdditionInProgress);

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  assert(positionEvent);

  mitk::BaseRenderer* renderer = positionEvent->GetSender();
  const mitk::PlaneGeometry* planeGeometry = renderer->GetCurrentWorldPlaneGeometry();
  mitk::Point2D position;
  planeGeometry->Map(positionEvent->GetPositionInWorld(), position);

  this->SetEraserPosition(position);
  renderer->RequestUpdate();


  int dataIndex = this->GetDataIndex(true);
  this->DoMouseMoved(action, event, dataIndex, 1, 0);
  return true;
}


//-----------------------------------------------------------------------------
bool PaintbrushTool::StopAddingAddition(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* event)
{
  assert(m_AddingAdditionInProgress);

  this->SetEraserVisible(false, event->GetSender());

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

  InteractionEventObserverMutex::GetInstance()->Unlock(this);
  m_AddingAdditionInProgress = false;

  return true;
}


//-----------------------------------------------------------------------------
bool PaintbrushTool::StartAddingSubtraction(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  m_AddingSubtractionInProgress = true;
  InteractionEventObserverMutex::GetInstance()->Lock(this);

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  assert(positionEvent);

  mitk::BaseRenderer* renderer = positionEvent->GetSender();
  const mitk::PlaneGeometry* planeGeometry = renderer->GetCurrentWorldPlaneGeometry();
  m_EraserCursor->SetPlaneGeometry(const_cast<mitk::PlaneGeometry*>(planeGeometry));
  mitk::Point2D position;
  planeGeometry->Map(positionEvent->GetPositionInWorld(), position);

  this->SetEraserPosition(position);
  this->SetEraserVisible(true, renderer);
  renderer->RequestUpdate();

  int dataIndex = this->GetDataIndex(false);
  return this->MarkInitialPosition(dataIndex, action, event);
}


//-----------------------------------------------------------------------------
bool PaintbrushTool::KeepAddingSubtraction(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  assert(m_AddingSubtractionInProgress);

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  assert(positionEvent);

  mitk::BaseRenderer* renderer = positionEvent->GetSender();
  const mitk::PlaneGeometry* planeGeometry = renderer->GetCurrentWorldPlaneGeometry();
  mitk::Point2D position;
  planeGeometry->Map(positionEvent->GetPositionInWorld(), position);

  this->SetEraserPosition(position);
  renderer->RequestUpdate();

  int dataIndex = this->GetDataIndex(false);
  this->DoMouseMoved(action, event, dataIndex, 1, 0);
  return true;
}


//-----------------------------------------------------------------------------
bool PaintbrushTool::StopAddingSubtraction(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* event)
{
  assert(m_AddingSubtractionInProgress);

  this->SetEraserVisible(false, event->GetSender());

  int dataIndex = this->GetDataIndex(false);
  this->SetInvalidRegion(dataIndex);
  this->SegmentationEdited.Send(dataIndex);

  InteractionEventObserverMutex::GetInstance()->Unlock(this);
  m_AddingSubtractionInProgress = false;

  return true;
}


//-----------------------------------------------------------------------------
bool PaintbrushTool::StartRemovingSubtraction(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  m_RemovingSubtractionInProgress = true;
  InteractionEventObserverMutex::GetInstance()->Lock(this);

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  assert(positionEvent);

  mitk::BaseRenderer* renderer = positionEvent->GetSender();
  const mitk::PlaneGeometry* planeGeometry = renderer->GetCurrentWorldPlaneGeometry();
  m_EraserCursor->SetPlaneGeometry(const_cast<mitk::PlaneGeometry*>(planeGeometry));
  mitk::Point2D position;
  planeGeometry->Map(positionEvent->GetPositionInWorld(), position);

  this->SetEraserPosition(position);
  this->SetEraserVisible(true, renderer);
  renderer->RequestUpdate();

  int dataIndex = this->GetDataIndex(false);
  return this->MarkInitialPosition(dataIndex, action, event);
}


//-----------------------------------------------------------------------------
bool PaintbrushTool::KeepRemovingSubtraction(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  assert(m_RemovingSubtractionInProgress);

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  assert(positionEvent);

  mitk::BaseRenderer* renderer = positionEvent->GetSender();
  const mitk::PlaneGeometry* planeGeometry = renderer->GetCurrentWorldPlaneGeometry();
  mitk::Point2D position;
  planeGeometry->Map(positionEvent->GetPositionInWorld(), position);

  this->SetEraserPosition(position);
  renderer->RequestUpdate();

  int dataIndex = this->GetDataIndex(false);
  this->DoMouseMoved(action, event, dataIndex, 0, 1);
  return true;
}


//-----------------------------------------------------------------------------
bool PaintbrushTool::StopRemovingSubtraction(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* event)
{
  assert(m_RemovingSubtractionInProgress);

  this->SetEraserVisible(false, event->GetSender());

  int dataIndex = this->GetDataIndex(false);
  this->SetInvalidRegion(dataIndex);
  this->SegmentationEdited.Send(dataIndex);

  InteractionEventObserverMutex::GetInstance()->Unlock(this);
  m_RemovingSubtractionInProgress = false;

  return true;
}


//-----------------------------------------------------------------------------
void PaintbrushTool::SetRegion(unsigned int dataIndex, bool valid, const std::vector<int>& boundingBox)
{
  mitk::DataNode* workingNode = m_ToolManager->GetWorkingData(dataIndex);
  assert(workingNode);

  // This property should always exist, as we create it when the volume is created.
  mitk::BaseProperty* baseProperty = workingNode->GetProperty(REGION_PROPERTY_NAME.c_str());
  ITKRegionParametersDataNodeProperty::Pointer prop = dynamic_cast<ITKRegionParametersDataNodeProperty*>(baseProperty);

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


//-----------------------------------------------------------------------------
void PaintbrushTool::SetInvalidRegion(unsigned int dataIndex)
{
  this->SetRegion(dataIndex, false);
}


//-----------------------------------------------------------------------------
void PaintbrushTool::SetValidRegion(unsigned int dataIndex, const std::vector<int>& boundingBox)
{
  this->SetRegion(dataIndex, true, boundingBox);
}


//-----------------------------------------------------------------------------
void PaintbrushTool::SetEraserVisible(bool visible, mitk::BaseRenderer* renderer)
{
  if (m_EraserVisible == visible)
  {
    return;
  }

  if (mitk::DataStorage* dataStorage = m_ToolManager->GetDataStorage())
  {
    if (visible)
    {
      dataStorage->Add(m_EraserCursorNode);
    }
    else
    {
      dataStorage->Remove(m_EraserCursorNode);
    }
  }

  if (visible && renderer)
  {
    const mitk::PlaneGeometry* planeGeometry = renderer->GetCurrentWorldPlaneGeometry();
    m_EraserCursor->SetPlaneGeometry(const_cast<mitk::PlaneGeometry*>(planeGeometry));
  }

  m_EraserCursorNode->SetVisibility(visible, renderer);
  m_EraserVisible = visible;

  if (renderer)
  {
    renderer->ForceImmediateUpdate();
  }
}


//-----------------------------------------------------------------------------
void PaintbrushTool::ExecuteOperation(mitk::Operation* operation)
{
  if (!operation) return;

  switch (operation->GetOperationType())
  {
  case MIDAS_PAINTBRUSH_TOOL_OP_EDIT_IMAGE:
    {
      PaintbrushToolOpEditImage *op = static_cast<PaintbrushToolOpEditImage*>(operation);
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


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void PaintbrushTool::RunITKProcessor(
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

}
