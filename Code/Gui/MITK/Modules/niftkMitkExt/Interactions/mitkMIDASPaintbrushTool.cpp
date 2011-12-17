/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-01 09:00:22 +0100 (Mon, 01 Aug 2011) $
 Revision          : $Revision: 6894 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASPaintbrushTool.h"
#include "mitkMIDASPaintbrushTool.xpm"
#include "vtkImageData.h"
#include "mitkDataStorageUtils.h"
#include "mitkVector.h"
#include "mitkToolManager.h"
#include "mitkBaseRenderer.h"
#include "mitkImageAccessByItk.h"
#include "mitkInstantiateAccessFunctions.h"
#include "mitkITKImageImport.h"
#include "mitkRenderingManager.h"
#include "mitkUndoController.h"

const std::string mitk::MIDASPaintbrushTool::EDITING_PROPERTY_NAME = std::string("midas.morph.editing");
const std::string mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_X = std::string("midas.morph.editing.index.x");
const std::string mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_Y = std::string("midas.morph.editing.index.y");
const std::string mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_Z = std::string("midas.morph.editing.index.z");
const std::string mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_X = std::string("midas.morph.editing.size.x");
const std::string mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_Y = std::string("midas.morph.editing.size.y");
const std::string mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_Z = std::string("midas.morph.editing.size.z");
const std::string mitk::MIDASPaintbrushTool::EDITING_PROPERTY_REGION_SET = std::string("midas.morph.editing.region.set");

const mitk::OperationType mitk::MIDASPaintbrushTool::OP_EDIT_IMAGE = 320410;

namespace mitk{
  MITK_TOOL_MACRO(NIFTKMITKEXT_EXPORT, MIDASPaintbrushTool, "MIDAS Paintbrush Tool");
}

mitk::OpEditImage::OpEditImage(
    mitk::OperationType type,
    bool redo,
    int imageNumber,
    unsigned char valueToWrite,
    mitk::Image* imageToEdit,
    mitk::DataNode* nodeToEdit,
    ProcessorType* processor
    )
: mitk::Operation(type)
, m_Redo(redo)
, m_ImageNumber(imageNumber)
, m_ValueToWrite(valueToWrite)
, m_ImageToEdit(imageToEdit)
, m_NodeToEdit(nodeToEdit)
, m_Processor(processor)
{
}

mitk::MIDASPaintbrushTool::MIDASPaintbrushTool() : MIDASContourTool("MIDASPaintbrushTool")
, m_CursorSize(1)
{
  CONNECT_ACTION( 320401, OnLeftMousePressed );
  CONNECT_ACTION( 320402, OnLeftMouseReleased );
  CONNECT_ACTION( 320403, OnLeftMouseMoved );
  CONNECT_ACTION( 320404, OnMiddleMousePressed );
  CONNECT_ACTION( 320405, OnMiddleMouseReleased );
  CONNECT_ACTION( 320406, OnMiddleMouseMoved );
  CONNECT_ACTION( 320407, OnRightMousePressed );
  CONNECT_ACTION( 320408, OnRightMouseReleased );
  CONNECT_ACTION( 320409, OnRightMouseMoved );

  m_Interface = new mitk::MIDASPaintbrushTool::MIDASPaintbrushToolEventInterface();
  m_Interface->SetMIDASPaintbrushTool( this );
}

mitk::MIDASPaintbrushTool::~MIDASPaintbrushTool()
{
  if (m_Interface)
  {
    m_Interface->Delete();
  }
}

const char* mitk::MIDASPaintbrushTool::GetName() const
{
  return "Paintbrush";
}

const char** mitk::MIDASPaintbrushTool::GetXPM() const
{
  return mitkMIDASPaintbrushTool_xpm;
}

void mitk::MIDASPaintbrushTool::GetListOfAffectedVoxels(
    const PlaneGeometry& planeGeometry,
    Point3D& currentPoint,
    Point3D& previousPoint,
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
  double stepSize = this->CalculateStepSize(spacing);

  mitk::Point3D mostRecentPoint = previousPoint;
  mitk::Point3D vectorDifference;
  mitk::Point3D projectedPointIn3DVoxels;
  mitk::Point3D cursorPointIn3DVoxels;
  mitk::Index3D affectedVoxel;

  this->GetDifference(currentPoint, mostRecentPoint, vectorDifference);
  double length = this->GetSquaredDistanceBetweenPoints(currentPoint, mostRecentPoint);

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

        for (int dimension = 0; dimension < 2; dimension++)
        {
          cursorPointIn3DVoxels = projectedPointIn3DVoxels;

          int actualCursorSize = m_CursorSize - 1;

          for (int offset = -actualCursorSize; offset <= actualCursorSize; offset++)
          {
            cursorPointIn3DVoxels[whichTwoAxesInVoxelSpace[dimension]] = projectedPointIn3DVoxels[whichTwoAxesInVoxelSpace[dimension]] + offset;

            for (int i = 0; i < 3; i++)
            {
              affectedVoxel[i] = (long int)cursorPointIn3DVoxels[i];
            }

            // Check we are not outside image
            if (m_WorkingImageGeometry->IsIndexInside(affectedVoxel))
            {
              processor.AddToList(affectedVoxel);
            }
          }
        }
      } // end for k, foreach step
    } // end if steps > 0
  } // end if length > 0
}

bool mitk::MIDASPaintbrushTool::MarkInitialPosition(Action* action, const StateEvent* stateEvent)
{
  // Don't forget to call baseclass method, which sets a few internal variables used for figuring out geometry, and image dimensions etc.
  if (!MIDASContourTool::OnMousePressed(action, stateEvent))
  {
    return false;
  }

  // Make sure we have a valid position event, otherwise no point continuing.
  const PositionEvent* positionEvent = dynamic_cast<const PositionEvent*>(stateEvent->GetEvent());
  if (!positionEvent)
  {
    return false;
  }

  // Set reference data, but we don't draw anything at this stage
  m_MostRecentPointInMillimetres = positionEvent->GetWorldPosition();

  return true;
}

bool mitk::MIDASPaintbrushTool::DoInitialCheck(Action* action, const StateEvent* stateEvent)
{
  if (!FeedbackContourTool::OnMouseMoved( action, stateEvent ))
  {
    return false;
  }

  if (m_WorkingImage == NULL || m_WorkingImageGeometry == NULL)
  {
    return false;
  }

  return true;
}

void mitk::MIDASPaintbrushTool::UpdateRegionSetProperty(int imageNumber, bool isRegionSet)
{
  this->UpdateWorkingImageBooleanProperty(imageNumber, mitk::MIDASPaintbrushTool::EDITING_PROPERTY_REGION_SET, isRegionSet);
}

void mitk::MIDASPaintbrushTool::UpdateEditingProperty(int imageNumber, bool editingPropertyValue)
{
  this->UpdateWorkingImageBooleanProperty(imageNumber, mitk::MIDASPaintbrushTool::EDITING_PROPERTY_NAME, editingPropertyValue);
}

bool mitk::MIDASPaintbrushTool::DoMouseMoved(Action* action,
    const StateEvent* stateEvent,
    int imageNumber,
    unsigned char valueForRedo,
    unsigned char valueForUndo

    )
{
  if (!this->DoInitialCheck(action, stateEvent))
  {
    return false;
  }

  const PositionEvent* positionEvent = dynamic_cast<const PositionEvent*>(stateEvent->GetEvent());
  if (!positionEvent)
  {
    return false;
  }

  const PlaneGeometry* planeGeometry( dynamic_cast<const PlaneGeometry*> (positionEvent->GetSender()->GetCurrentWorldGeometry2D() ) );
  if ( !planeGeometry )
  {
    return false;
  }

  DataNode* workingNode( m_ToolManager->GetWorkingData(imageNumber) );
  assert(workingNode);

  mitk::Image::Pointer imageToWriteTo = static_cast<mitk::Image*>(workingNode->GetData());
  assert(imageToWriteTo);

  mitk::Point3D currentPoint = positionEvent->GetWorldPosition();

  ProcessorType::Pointer processor = ProcessorType::New();
  this->GetListOfAffectedVoxels(*planeGeometry, currentPoint, m_MostRecentPointInMillimetres, (*processor));

  if (processor->GetNumberOfVoxels() > 0)
  {
    MITK_DEBUG << "DoMouseMoved::cp=" << currentPoint << ", pp=" << m_MostRecentPointInMillimetres << ", vox=" << processor->GetNumberOfVoxels() << std::endl;

    try
    {

      this->UpdateRegionSetProperty(imageNumber, false);

      std::vector<int> boundingBox = processor->ComputeMinimalBoundingBox();
      MITK_DEBUG << "DoMouseMoved: boundingBox=" << boundingBox[0] << ", " << boundingBox[1] << ", " << boundingBox[2] << ", " << boundingBox[3] << ", " << boundingBox[4] << ", " << boundingBox[5] << std::endl;

      OpEditImage *doOp = new OpEditImage(OP_EDIT_IMAGE, true, imageNumber, valueForRedo, imageToWriteTo, workingNode, processor);
      OpEditImage *undoOp = new OpEditImage(OP_EDIT_IMAGE, false, imageNumber, valueForUndo, imageToWriteTo, workingNode, processor);
      mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Edit Image");
      mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );

      ExecuteOperation(doOp);

      // These are used so that the ITK pipeline in MIDSAMorphologicalSegmentorView knows which region to copy and update.
      workingNode->ReplaceProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_X.c_str(), mitk::IntProperty::New(boundingBox[0]));
      workingNode->ReplaceProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_Y.c_str(), mitk::IntProperty::New(boundingBox[1]));
      workingNode->ReplaceProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_Z.c_str(), mitk::IntProperty::New(boundingBox[2]));
      workingNode->ReplaceProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_X.c_str(), mitk::IntProperty::New(boundingBox[3]));
      workingNode->ReplaceProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_Y.c_str(), mitk::IntProperty::New(boundingBox[4]));
      workingNode->ReplaceProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_Z.c_str(), mitk::IntProperty::New(boundingBox[5]));

      this->UpdateRegionSetProperty(imageNumber, true);
    }
    catch( itk::ExceptionObject & err )
    {
      MITK_ERROR << "Failed to perform edit in mitkMIDASPaintrushTool due to:" << err << std::endl;
    }
  }

  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
  m_MostRecentPointInMillimetres = currentPoint;
  return true;
}

bool mitk::MIDASPaintbrushTool::OnLeftMousePressed (Action* action, const StateEvent* stateEvent)
{
  return this->MarkInitialPosition(action, stateEvent);
}

bool mitk::MIDASPaintbrushTool::OnLeftMouseMoved(Action* action, const StateEvent* stateEvent)
{
  this->UpdateEditingProperty(1, true);
  this->DoMouseMoved(action, stateEvent, 1, 255, 0);
  return true;
}

bool mitk::MIDASPaintbrushTool::OnLeftMouseReleased(Action* action, const StateEvent* stateEvent)
{
  this->UpdateEditingProperty(1, false);
  return true;
}

bool mitk::MIDASPaintbrushTool::OnMiddleMousePressed (Action* action, const StateEvent* stateEvent)
{
  return this->MarkInitialPosition(action, stateEvent);
}

bool mitk::MIDASPaintbrushTool::OnMiddleMouseMoved(Action* action, const StateEvent* stateEvent)
{
  this->UpdateEditingProperty(0, true);
  this->DoMouseMoved(action, stateEvent, 0, 255, 0);
  return true;
}

bool mitk::MIDASPaintbrushTool::OnMiddleMouseReleased(Action* action, const StateEvent* stateEvent)
{
  this->UpdateEditingProperty(0, false);
  return true;
}

bool mitk::MIDASPaintbrushTool::OnRightMousePressed (Action* action, const StateEvent* stateEvent)
{
  return this->MarkInitialPosition(action, stateEvent);
}

bool mitk::MIDASPaintbrushTool::OnRightMouseMoved(Action* action, const StateEvent* stateEvent)
{
  this->UpdateEditingProperty(0, true);
  this->DoMouseMoved(action, stateEvent, 0, 0, 255);
  return true;
}

bool mitk::MIDASPaintbrushTool::OnRightMouseReleased(Action* action, const StateEvent* stateEvent)
{
  this->UpdateEditingProperty(0, false);
  return true;
}

void mitk::MIDASPaintbrushTool::ExecuteOperation(Operation* operation)
{
  if (!operation) return;

  switch (operation->GetOperationType())
  {
  case OP_EDIT_IMAGE:
    {
      OpEditImage *op = static_cast<OpEditImage*>(operation);
      unsigned char valueToWrite = op->GetValueToWrite();
      ProcessorType::Pointer processor = op->GetProcessor();
      mitk::Image::Pointer imageToEdit = op->GetImageToEdit();
      mitk::DataNode::Pointer nodeToEdit = op->GetNodeToEdit();
      bool redo = op->IsRedo();

      typedef mitk::ImageToItk< ImageType > ImageToItkType;
      ImageToItkType::Pointer imageToEditToItk = ImageToItkType::New();
      imageToEditToItk->SetInput(imageToEdit);
      imageToEditToItk->Update();

      RunITKProcessor<mitk::Tool::DefaultSegmentationDataType, 3>(imageToEditToItk->GetOutput(), processor, redo, valueToWrite);

      imageToEdit->Modified();
      nodeToEdit->Modified();

      mitk::RenderingManager::GetInstance()->RequestUpdateAll();
      break;
    }
  default:;
  }
}

template<typename TPixel, unsigned int VImageDimension>
void mitk::MIDASPaintbrushTool::RunITKProcessor(
    itk::Image<TPixel, VImageDimension>* itkImage,
    ProcessorType::Pointer processor,
    bool redo,
    unsigned char valueToWrite
    )
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;
  typedef typename ImageType::IndexType  IndexType;
  typedef typename ImageType::SizeType   SizeType;

  std::vector<int> boundingBox = processor->ComputeMinimalBoundingBox();

  IndexType regionIndex;
  regionIndex[0] = boundingBox[0];
  regionIndex[1] = boundingBox[1];
  regionIndex[2] = boundingBox[2];

  SizeType regionSize;
  regionSize[0] = boundingBox[3];
  regionSize[1] = boundingBox[4];
  regionSize[2] = boundingBox[5];

  RegionType roi;
  roi.SetIndex(regionIndex);
  roi.SetSize(regionSize);

  processor->SetDestinationImage(itkImage);
  processor->SetDestinationRegionOfInterest(roi);
  processor->SetValue(valueToWrite);

  if (redo)
  {
    processor->Redo();
  }
  else
  {
    processor->Undo();
  }
}
