/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-19 22:31:43 +0000 (Sat, 19 Nov 2011) $
 Revision          : $Revision: 7815 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASDrawTool.h"
#include "mitkMIDASDrawTool.xpm"
#include "mitkVector.h"
#include "mitkToolManager.h"
#include "mitkBaseRenderer.h"
#include "vtkImageData.h"


namespace mitk{
  MITK_TOOL_MACRO(NIFTKMITKEXT_EXPORT, MIDASDrawTool, "MIDAS Draw Tool");
}

mitk::MIDASDrawTool::MIDASDrawTool() : MIDASContourTool("MIDASDrawTool")
{
  // great magic numbers, connecting interactor straight to method calls.
  CONNECT_ACTION( 80, OnLeftMousePressed );
  CONNECT_ACTION( 42, OnLeftMouseReleased );
  CONNECT_ACTION( 90, OnLeftMouseMoved );
}

mitk::MIDASDrawTool::~MIDASDrawTool()
{

}

const char* mitk::MIDASDrawTool::GetName() const
{
  return "Draw";
}

const char** mitk::MIDASDrawTool::GetXPM() const
{
  return mitkMIDASDrawTool_xpm;
}

/**
 To start a contour, we initialise the "FeedbackCountour", which is the "Current" contour,
 and also store the current point, at which the mouse was pressed down. It's the next
 method OnMouseMoved that starts to draw the line.
*/
bool mitk::MIDASDrawTool::OnLeftMousePressed (Action* action, const StateEvent* stateEvent)
{
  // Don't forget to call baseclass method.
  MIDASContourTool::OnMousePressed(action, stateEvent);

  // Make sure we have a valid position event, otherwise no point continuing.
  const PositionEvent* positionEvent = dynamic_cast<const PositionEvent*>(stateEvent->GetEvent());
  if (!positionEvent) return false;

  // Initialize contours, and set properties.
  Contour* feedbackContour = FeedbackContourTool::GetFeedbackContour();
  feedbackContour->Initialize();
  feedbackContour->SetClosed(m_ContourClosed);
  feedbackContour->SetWidth(m_ContourWidth);

  // Initialize contours, and set properties.
  Contour* backgroundContour = MIDASContourTool::GetBackgroundContour();
  backgroundContour->Initialize();
  backgroundContour->SetClosed(m_ContourClosed);
  backgroundContour->SetWidth(m_ContourWidth);

  // Turn the feedback contours on, background contours off and default the colours.
  FeedbackContourTool::SetFeedbackContourVisible(true);
  FeedbackContourTool::SetFeedbackContourColorDefault();
  MIDASContourTool::SetBackgroundContourVisible(false);
  MIDASContourTool::SetBackgroundContourColorDefault();
  MIDASContourTool::SetCumulativeFeedbackContoursVisible(true);
  MIDASContourTool::SetCumulativeFeedbackContoursColorDefault();

  // Set reference data, but we don't draw anything at this stage
  m_MostRecentPointInMillimetres = positionEvent->GetWorldPosition();

  return true;
}

/**
 * When we finish a contour, we take the Current contour, and add it to the Cumulative contour.
 */
bool mitk::MIDASDrawTool::OnLeftMouseReleased(Action* action, const StateEvent* stateEvent)
{
  // Make sure we have a valid position event, otherwise no point continuing.
  const PositionEvent* positionEvent = dynamic_cast<const PositionEvent*>(stateEvent->GetEvent());
  if (!positionEvent) return false;

  /**
   * When the mouse is released, we add the current contour to the cumulative one,
   * and reset the current one to being invisible. When a new contour is started it
   * the current feedback contour will be reset to have zero length.
   */
  Contour* feedbackContour = FeedbackContourTool::GetFeedbackContour();
  Contour* backgroundContour = MIDASContourTool::GetBackgroundContour();
  this->AddToCumulativeFeedbackContours(*feedbackContour, *backgroundContour);

  // Turn the current contour off. It gets initialised to zero length the next time the user clicks.
  FeedbackContourTool::SetFeedbackContourVisible(false);
  MIDASContourTool::SetBackgroundContourVisible(false);

  // Set the colour of the feedback, to make sure the new points that were just added are rendered correctly.
  MIDASContourTool::SetCumulativeFeedbackContoursColorDefault();

  // Set this flag to indicate that we have stopped editing, which will trigger an update of the region growing.
  this->UpdateWorkingImageBooleanProperty(0, mitk::MIDASContourTool::EDITING_PROPERTY_NAME, false);

  // Make sure all views everywhere get updated.
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
  return true;
}

/**
 As the mouse is moved, we draw a line in 2D slice, round edges of voxels.
 The complexity lies in the fact that MouseMove events don't give you every
 pixel (unless you move your mouse slowly), so you have to draw a line between
 two points that may span more than one voxel, or fractions of a voxel.
*/
bool mitk::MIDASDrawTool::OnLeftMouseMoved(Action* action, const StateEvent* stateEvent)
{
  if (!FeedbackContourTool::OnMouseMoved( action, stateEvent )) return false;

  if (m_WorkingImage == NULL || m_WorkingImageGeometry == NULL) return false;

  const PositionEvent* positionEvent = dynamic_cast<const PositionEvent*>(stateEvent->GetEvent());
  if (!positionEvent) return false;

  const PlaneGeometry* planeGeometry( dynamic_cast<const PlaneGeometry*> (positionEvent->GetSender()->GetCurrentWorldGeometry2D() ) );
  if ( !planeGeometry ) return false;

  // Retrieve the contour that we will add points to.
  Contour* feedbackContour = FeedbackContourTool::GetFeedbackContour();
  Contour* backgroundContour = MIDASContourTool::GetBackgroundContour();

  // Draw lines between the current pixel position, and the previous one (stored in OnMousePressed).
  unsigned int numberAdded = this->DrawLineAroundVoxelEdges
                             (
                               *m_WorkingImage,
                               *m_WorkingImageGeometry,
                               *planeGeometry,
                                positionEvent->GetWorldPosition(),
                                m_MostRecentPointInMillimetres,
                               *feedbackContour,
                               *backgroundContour
                             );

  // This gets updated as the mouse moves, so that each new segement of line gets added onto the previous.
  if (numberAdded > 0)
  {
    m_MostRecentPointInMillimetres = positionEvent->GetWorldPosition();
  }

  // Set this flag to indicate that we are editing, which will block the update of the region growing.
  this->UpdateWorkingImageBooleanProperty(0, mitk::MIDASContourTool::EDITING_PROPERTY_NAME, true);

  // Make sure all views everywhere get updated.
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
  return true;
}

void mitk::MIDASDrawTool::Wipe()
{
  mitk::MIDASContourTool::Wipe();
}

