/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKMIDASPOLYTOOL_H
#define MITKMIDASPOLYTOOL_H

#include "niftkMIDASExports.h"
#include "mitkMIDASContourTool.h"
#include "mitkMIDASPolyToolEventInterface.h"
#include <mitkPointSet.h>
#include <mitkOperation.h>
#include <mitkOperationActor.h>

namespace mitk {

  /**
   * \class MIDASPolyTool
   * \brief Tool to draw poly lines around voxel edges like MIDAS does rather than through them as most of the MITK tools do.
   *
   * Provides
   * <pre>
   * 1. Left mouse button = place marker for poly line
   * 2. Middle mouse button = select nearest marker point
   * 3. Move with middle mouse button down = move/drag the marker point selected in step 2.
   * </pre>
   * and includes Undo/Redo functionality. The poly lines keep going until the tool is deselected.
   * When the tool is deselected, the poly line is copied to the mitk::ToolManagers WorkingData, specifically dataset 2,
   * which should be the mitk::ContourSet representing the current set of contours in the gui, which in MIDAS terms
   * is the green lines representing the current segmentation.
   */
  class NIFTKMIDAS_EXPORT MIDASPolyTool : public MIDASContourTool {

  public:

    mitkClassMacro(MIDASPolyTool, Tool);
    itkNewMacro(MIDASPolyTool);

    /// \see mitk::Tool::GetName()
    virtual const char* GetName() const;

    /// \see mitk::Tool::GetXPM()
    virtual const char** GetXPM() const;

    /// \brief We store the name of the anchor points node, which is used to store the first point on a poly line.
    static const std::string MIDAS_POLY_TOOL_ANCHOR_POINTS;

    /// \brief We store the name of the previous contour node, which is the contour display in green when middle-click dragging the poly line.
    static const std::string MIDAS_POLY_TOOL_PREVIOUS_CONTOUR;

    /// \brief Method to enable this class to interact with the Undo/Redo framework.
    virtual void ExecuteOperation(Operation* operation);

    /// \brief When called, we initialize contours, as the PolyLine keeps going until the whole tool is Activated/Deactivated.
    virtual void Activated();

    /// \brief When called, add the current poly line to the node specified by mitk::MIDASTool::CURRENT_CONTOURS_NAME.
    virtual void Deactivated();

    /// \see mitk::StateMachine::CanHandleEvent
    float CanHandleEvent(const StateEvent *) const;

    /// \brief When called, we incrementally build up a poly line.
    virtual bool OnLeftMousePressed(Action* action, const StateEvent* stateEvent);

    /// \brief When called, we select the closest point in poly line, ready to move it.
    virtual bool OnMiddleMousePressed(Action* action, const StateEvent* stateEvent);

    /// \brief When called, we move the selected point and hence move the poly line.
    virtual bool OnMiddleMousePressedAndMoved(Action* action, const StateEvent* stateEvent);

    /// \brief When called, we release the selected point and hence stop moving the poly line.
    virtual bool OnMiddleMouseReleased(Action* action, const StateEvent* stateEvent);

    /// \brief Clears the contour, meaning it re-initialised the feedback contour in
    /// mitk::FeedbackContourTool, and also the background contour in mitk::MIDASContourTool
    /// and the Previous Contour and Poly Line points in this class.
    virtual void ClearData();

  protected:

    MIDASPolyTool(); // purposely hidden
    virtual ~MIDASPolyTool(); // purposely hidden

  private:

    /// \brief Sets the m_PolyLinePointSet to be visible/invisible.
    void SetPolyLinePointSetVisible(bool visible);

    /// \brief Sets whether the m_PreviousContour is visible/invisible.
    void SetPreviousContourVisible(bool visible);

    /// \brief Makes sure the previous contour is not rendered in any 3D window.
    void Disable3dRenderingOfPreviousContour();

    /// \brief Takes the contourReferencePointsInput and planeGeometry, and if there are >1 points in the contour, generates new feedbackContour and backgroundContour by calling mitk::MIDASContourTool::DrawLineAroundVoxelEdges.
    void DrawWholeContour(const mitk::Contour& contourReferencePointsInput, const PlaneGeometry& planeGeometry, mitk::Contour& feedbackContour, mitk::Contour& backgroundContour);

    /// \brief Called from OnMiddleMousePressed and OnMiddleMousePressedAndMoved, used to draw the previous contour in green, and the current contour (which is being dragged by the mouse with the middle click) in yellow.
    void UpdateContours(Action* action, const StateEvent* stateEvent, bool provideUndo, bool registerNewPoint);

    /// \brief Called from UpdateContours, takes the given point and geometry, and the existing contour (poly line), and calculates the closest point in the current contourReferencePointsInput, sets it to the closestCornerPoint and redraws the feedbackContour and backgroundContour by calling DrawWholeContour.
    void UpdateFeedbackContour(bool registerNewPoint, const mitk::Point3D& closestCornerPoint, const PlaneGeometry& planeGeometry, mitk::Contour& contourReferencePointsInput, mitk::Contour& feedbackContour, mitk::Contour& backgroundContour, bool provideUndo);

    /// \brief We use this to store the last point between mouse clicks.
    mitk::Point3D m_MostRecentPointInMillimetres;

    /// \brief Reference points are points containing just the nodes that were clicked.
    mitk::Contour::Pointer m_ReferencePoints;

    /// \brief When we middle-click-and-drag, we need to remember where the previous line was, so we can draw it in green.
    mitk::Contour::Pointer m_PreviousContourReferencePoints;

    /// \brief When user moves the contour, we give interactive feedback of
    /// the "Current" contour in yellow, and the "Previous" contour in green.
    mitk::Contour::Pointer  m_PreviousContour;
    mitk::DataNode::Pointer m_PreviousContourNode;
    bool                    m_PreviousContourVisible;

    /// \brief Use this point set to render a single seed position as a cross, for the start of the current contour.
    mitk::PointSet::Pointer m_PolyLinePointSet;
    mitk::DataNode::Pointer m_PolyLinePointSetNode;
    bool                    m_PolyLinePointSetVisible;

    /// \brief Operation constant, used in Undo/Redo framework.
    static const mitk::OperationType MIDAS_POLY_TOOL_OP_ADD_TO_FEEDBACK_CONTOUR;
    static const mitk::OperationType MIDAS_POLY_TOOL_OP_UPDATE_FEEDBACK_CONTOUR;

    /// \brief Pointer to interface object, used as callback in Undo/Redo framework
    MIDASPolyToolEventInterface::Pointer m_Interface;

    /// \brief When we are dragging a point, we want to keep track of the point we clicked on,
    /// and not always pick the closest point. Otherwise, if you middle clicked on a point
    /// and move the cursor around, and approach another point while dragging, the chosen point will swap.
    unsigned int m_DraggedPointIndex;

  };//class


}//namespace

#endif
