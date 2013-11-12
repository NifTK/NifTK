/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASContourTool_h
#define mitkMIDASContourTool_h

#include "niftkMIDASExports.h"
#include "mitkMIDASContourToolEventInterface.h"
#include "mitkMIDASTool.h"
#include <mitkPointUtils.h>
#include <mitkContourModel.h>
#include <mitkContourModelSet.h>
#include <mitkOperation.h>
#include <mitkOperationActor.h>
#include <mitkMessage.h>

namespace mitk {

/**
 * \class MIDASContourTool
 * \brief Provides common functionality for mitk::MIDASDrawTool and mitk::MIDASPolyTool
 * where these two tools enable drawing lines and poly-lines around voxel edges.
 *
 * This class derives from mitk::FeedbackContourTool, and uses several contours to
 * do its magic.  The base class "FeedbackContour" is the one that is visible as the tool
 * is used. In addition, in this class we store a "BackgroundContour". The FeedbackContour
 * goes round the edges of each voxel, and the BackgroundContour simply stores each
 * mouse position, as each mouse event is received, and hence contains the trajectory of the
 * mouse.
 *
 * \sa mitk::FeedbackContourTool
 * \sa mitk::MIDASTool
 * \sa mitk::MIDASDrawTool
 * \sa mitk::MIDASPolyTool
 */
class NIFTKMIDAS_EXPORT MIDASContourTool : public MIDASTool {

public:

  mitkClassMacro(MIDASContourTool, MIDASTool);

  /// \brief We store the name of a property to say we are editing.
  static const std::string EDITING_PROPERTY_NAME;

  /// \brief We store the name of the background contour, which is the contour storing exact mouse position events.
  static const std::string MIDAS_CONTOUR_TOOL_BACKGROUND_CONTOUR;

  /// \brief Method to enable this class to interact with the Undo/Redo framework.
  virtual void ExecuteOperation(Operation* operation);

  /// \brief Clears the contour, meaning it re-initialised the feedback contour in mitk::FeedbackContourTool, and also the background contour herein.
  virtual void ClearData();

  /// \brief Get a pointer to the current feedback contour.
  virtual mitk::ContourModel* GetContour();

  /// \brief Turns the feedback contour on/off.
  virtual void SetFeedbackContourVisible(bool);

  /// \brief Copies contour from a to b.
  static void CopyContour(mitk::ContourModel &a, mitk::ContourModel &b);

  /// \brief Copies contour set from a to b.
  static void CopyContourSet(mitk::ContourModelSet &a, mitk::ContourModelSet &b, bool initialise=true);

  /// \brief Initialises the output contour b with properties like, closed, width and selected, copied from the reference contour a.
  static void InitialiseContour(mitk::ContourModel &a, mitk::ContourModel &b);

  /// \brief Used to signal that the contours have changed.
  Message<> ContoursHaveChanged;

protected:

  MIDASContourTool(); // purposely hidden
  MIDASContourTool(const char* type); // purposely hidden
  virtual ~MIDASContourTool(); // purposely hidden

  /// \brief Calls the FeedbackContour::OnMousePressed method, then checks for working image, reference image and geometry.
  virtual bool OnMousePressed (Action*, const StateEvent*);

  /// \brief This method makes sure that the argument node will not show up in ANY 3D viewer thats currently registered with the global mitk::RenderingManager.
  void Disable3dRenderingOfNode(mitk::DataNode* node);

  /// \brief Adds the given contour to the Working Data registered with mitk::ToolManager, where the ToolManager can have multiple data sets registered, so we add the contour to the dataset specified by dataSetNumber.
  void AccumulateContourInWorkingData(mitk::ContourModel& contour, int dataSetNumber);

  // Utility methods for helping draw lines that require m_Geometry to be set.
  void ConvertPointToVoxelCoordinate(const mitk::Point3D& inputInMillimetreCoordinates, mitk::Point3D& outputInVoxelCoordinates);
  void ConvertPointToNearestVoxelCentre(const mitk::Point3D& inputInMillimetreCoordinates, mitk::Point3D& outputInVoxelCoordinates);
  void ConvertPointToNearestVoxelCentreInMillimetreCoordinates(const mitk::Point3D& inputInMillimetreCoordinates, mitk::Point3D& outputInMillimetreCoordinates);
  void GetClosestCornerPoint2D(const mitk::Point3D& pointInMillimetreCoordinate, int* whichTwoAxesInVoxelSpace, mitk::Point3D& cornerPointInMillimetreCoordinates);
  bool AreDiagonallyOpposite(const mitk::Point3D& cornerPointInMillimetreCoordinates1, const mitk::Point3D& cornerPointInMillimetreCoordinates2, int* whichTwoAxesInVoxelSpace);
  void GetAdditionalCornerPoint(const mitk::Point3D& cornerPoint1InMillimetreCoordinates, const mitk::Point3D& point2InMillimetreCoordinates, const mitk::Point3D& cornerPoint2InMillimetreCoordinates, int* whichTwoAxesInVoxelSpace, mitk::Point3D& outputInMillimetreCoordinates);

  // Main method for drawing a line:
  //   1.) from previousPoint to currentPoint working around voxel corners, output in contourAroundCorners
  //   2.) from previousPoint to currentPoint working in a straight line, output in contourAlongLine
  // Returns the number of points added
  unsigned int DrawLineAroundVoxelEdges(
      const mitk::Image& image,                 // input
      const mitk::Geometry3D& geometry3D,       // input
      const mitk::PlaneGeometry& planeGeometry, // input
      const mitk::Point3D& currentPoint,        // input
      const mitk::Point3D& previousPoint,       // input
      mitk::ContourModel& contourAroundCorners,      // output
      mitk::ContourModel& contourAlongLine           // output
      );

  // Methods for manipulating the "BackgroundContour", which typically doesn't get drawn, but is useful for converting to image coordinates, e.g. for rendering into images for boundaries.
  mitk::ContourModel* GetBackgroundContour();
  void SetBackgroundContour(mitk::ContourModel&);
  void Disable3dRenderingOfBackgroundContour();
  void SetBackgroundContourVisible(bool);
  void SetBackgroundContourColor( float r, float g, float b );
  void SetBackgroundContourColorDefault();

  // We default this to 1, and use throughout.
  int m_ContourWidth;

  // We default this to false, and use throughout.
  bool m_ContourClosed;

  // We default this to 0.01, and use throughout when comparing point positions.
  float m_Tolerance;

  // This is the 3D geometry associated with the m_WorkingImage
  mitk::Geometry3D* m_WorkingImageGeometry;

  // This is the current 3D working image (the image that is the segmentation, i.e. a binary image)
  mitk::Image* m_WorkingImage;

  // This is the current 3D reference image (the image that is being segmented, i.e. a grey scale image)
  mitk::Image* m_ReferenceImage;

  // Like the base class mitkFeedbackContourTool, we keep a contour that is the straight line, exaclty as we iterate, not working around voxel corners.
  mitk::ContourModel::Pointer  m_BackgroundContour;
  mitk::DataNode::Pointer m_BackgroundContourNode;
  bool                    m_BackgroundContourVisible;

private:

  // Operation constant, used in Undo/Redo framework.
  static const mitk::OperationType MIDAS_CONTOUR_TOOL_OP_ACCUMULATE_CONTOUR;

  /// \brief Pointer to interface object, used as callback in Undo/Redo framework
  MIDASContourToolEventInterface::Pointer m_Interface;

};//class

}//namespace

#endif
