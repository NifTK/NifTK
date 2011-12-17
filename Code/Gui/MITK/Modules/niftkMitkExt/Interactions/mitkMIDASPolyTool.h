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
#ifndef MITKMIDASPOLYTOOL_H
#define MITKMIDASPOLYTOOL_H

#include "niftkMitkExtExports.h"
#include "mitkMIDASContourTool.h"
#include "mitkPointSet.h"

namespace mitk {

  /**
   * \class MIDASPolyTool
   * \brief Tool to draw poly lines around voxel edges rather than through them.
   */
  class NIFTKMITKEXT_EXPORT MIDASPolyTool : public MIDASContourTool {

  public:

    mitkClassMacro(MIDASPolyTool, Tool);
    itkNewMacro(MIDASPolyTool);

    virtual const char* GetName() const;
    virtual const char** GetXPM() const;

    // When called, we initialize contours, as the PolyLine keeps going until the whole tool is Activated/Deactivated.
    virtual void Activated();

    // When called, add the current poly line to the cumulative contours.
    virtual void Deactivated();

    // When called, we incrementally build up a poly line.
    virtual bool OnLeftMousePressed(Action* action, const StateEvent* stateEvent);

    // When called, we select the closest point in poly line, ready to move it.
    virtual bool OnMiddleMousePressed(Action* action, const StateEvent* stateEvent);

    // When called, we move the selected point and hence move the poly line.
    virtual bool OnMiddleMousePressedAndMoved(Action* action, const StateEvent* stateEvent);

    // When called, we release the selected point and hence stop moving the poly line.
    virtual bool OnMiddleMouseReleased(Action* action, const StateEvent* stateEvent);

  protected:

    MIDASPolyTool(); // purposely hidden
    virtual ~MIDASPolyTool(); // purposely hidden

  private:

    void SetPolyLinePointSetVisible(bool visible);
    void SetPreviousContourVisible(bool visible);
    void Disable3dRenderingOfPreviousContour();
    void DrawWholeContour(const mitk::Contour& contourReferencePointsInput, const PlaneGeometry& planeGeometry, mitk::Contour& feedbackContour, mitk::Contour& backgroundContour);
    void UpdateFeedbackContour(const mitk::Point3D& closestCornerPoint, const PlaneGeometry& planeGeometry, mitk::Contour& contourReferencePointsInput, mitk::Contour& feedbackContour, mitk::Contour& backgroundContour);
    void UpdateContours(Action* action, const StateEvent* stateEvent);

    // We use this to store the last point between mouse clicks.
    mitk::Point3D m_MostRecentPointInMillimetres;

    // Reference points are contours containing just the nodes that were clicked.
    // i.e. they represent the control points that define the poly line.
    mitk::Contour::Pointer m_ReferencePoints;
    mitk::Contour::Pointer m_PreviousContourReferencePoints;

    // Use this point set to render a single seed position, for the start of the current contour.
    mitk::PointSet::Pointer m_PolyLinePointSet;
    mitk::DataNode::Pointer m_PolyLinePointSetNode;
    bool                    m_PolyLinePointSetVisible;

    // When user moves the contour, we give interactive feedback of
    // the "Current" contour in yellow, and the "Previous" contour in green.
    // This flag controls whether the previous contour is visible,
    // and the data node is used to add it to the data manager.
    mitk::Contour::Pointer  m_PreviousContour;
    mitk::DataNode::Pointer m_PreviousContourNode;
    bool                    m_PreviousContourVisible;

  };//class


}//namespace

#endif
