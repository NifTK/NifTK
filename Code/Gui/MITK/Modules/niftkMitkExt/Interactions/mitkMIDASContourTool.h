/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-21 08:53:21 +0100 (Wed, 21 Sep 2011) $
 Revision          : $Revision: 7344 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MITKMIDASCONTOURTOOL_H
#define MITKMIDASCONTOURTOOL_H

#include "niftkMitkExtExports.h"
#include "mitkPointUtils.h"
#include "mitkMIDASTool.h"
#include "mitkExtractImageFilter.h"

namespace mitk {

  class NIFTKMITKEXT_EXPORT MIDASContourTool : public MIDASTool {

  public:

    mitkClassMacro(MIDASContourTool, MIDASTool);

    // Wipe's any tool specific data, such as contours, seed points etc.
    virtual void Wipe();

    // Base class method that checks a few things and sets m_Image and m_Geometry.
    virtual bool OnMousePressed (Action*, const StateEvent*);

    /// Make these typedefs available.
    typedef std::pair<mitk::Contour::Pointer, mitk::Contour::Pointer> PairOfContours;
    typedef std::pair<mitk::DataNode::Pointer, mitk::DataNode::Pointer> PairOfNodes;

    /// Return a pointer to the cumulative contours.
    const std::vector<PairOfContours>* GetCumulativeContours() { return &m_CumulativeFeedbackContours; }

    // We store a string of a property to say we are editing.
    static const std::string EDITING_PROPERTY_NAME;

  protected:

    MIDASContourTool(); // purposely hidden
    MIDASContourTool(const char* type); // purposely hidden
    virtual ~MIDASContourTool(); // purposely hidden

    // This method makes sure that the contour will not show up in ANY 3D viewer (thats currently registered).
    void Disable3dRenderingOfContour(mitk::DataNode* node);

    // I wrote a copy method because the assignment operator didn't appear to copy anything,
    void CopyContour(mitk::Contour& a, mitk::Contour& b);

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
        mitk::Contour& contourAroundCorners,      // output
        mitk::Contour& contourAlongLine           // output
        );

    // Methods for manipulating the "BackgroundContour", which typically doesn't get drawn, but is useful for converting to image coordinates, e.g. for rendering into images for boundaries.
    Contour* GetBackgroundContour();
    void SetBackgroundContour(Contour&);
    void Disable3dRenderingOfBackgroundContour(); // According to MITK commments, it appears there is a problem rendering contours in 3D
    void SetBackgroundContourVisible(bool);
    void SetBackgroundContourColor( float r, float g, float b );
    void SetBackgroundContourColorDefault();

    // Methods for manipulating the "Cumulative Contour", which increases as you draw more and more lines.
    void AddToCumulativeFeedbackContours(mitk::Contour& feedbackContour, mitk::Contour& backgroundContour);
    void ClearCumulativeFeedbackContours();
    void Disable3dRenderingOfCumulativeFeedbackContours(); // According to MITK commments, it appears there is a problem rendering contours in 3D
    void SetCumulativeFeedbackContoursVisible(bool);
    void SetCumulativeFeedbackContoursColor( float r, float g, float b );
    void SetCumulativeFeedbackContoursColorDefault();

    // Each time a contour is finished:
    //   (i.e. mitkMIDASPolyTool mouse clicked, or mitkMIDASDrawTool mouse released - we need a PositionEvent)
    //   we re-calculate the image of rendered lines.
    void UpdateImageOfRenderedContours();

    // We default this to 1, and use throughout.
    int m_ContourWidth;

    // We default this to false, and use throughout.
    bool m_ContourClosed;

    // We default this to 0.0001, and use throughout when comparing point positions.
    float m_Tolerance;

    // This is the 3D geometry associated with the m_WorkingImage
    mitk::Geometry3D* m_WorkingImageGeometry;

    // This is the current 3D working image (the image that is the segmentation, i.e. a binary image)
    mitk::Image* m_WorkingImage;

    // This is the current 3D reference image (the image that is being segmented, i.e. a grey scale image)
    mitk::Image* m_ReferenceImage;

    // Like the base class mitkFeedbackContourTool, we keep a contour that is the straight line, exaclty as we iterate, not working around voxel corners.
    mitk::Contour::Pointer  m_BackgroundContour;
    mitk::DataNode::Pointer m_BackgroundContourNode;
    bool                    m_BackgroundContourVisible;

    // We also keep matched pairs of:
    //   FeedBack contours, which actually get drawn, and go round voxel edge.
    //   Background contours, which don't get drawn, but go in straight line.
    // we have cumulative contours, which are a vector of contours.
    bool                           m_CumulativeFeedbackContoursVisible;
    std::vector<PairOfContours>    m_CumulativeFeedbackContours;
    std::vector<PairOfNodes>       m_CumulativeFeedbackContoursNodes;

  private:

  };//class

}//namespace

#endif
