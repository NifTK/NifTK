/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkProjectCameraRays_h
#define mitkProjectCameraRays_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <cv.h>

namespace mitk {

/**
 * \class ProjectCameraRays
 * \brief Takes a camera calibration (intrinsic and distorion) and projected every image pixel (default 1920 x 540)
 * to a ray from the camera origin (0,0,0) to a distance of 500 by default. Optionally include the camera to world transform.
 * Output is 1036800 rays
 * Currently the image that you are dealing with and hence the 2D pixel coordinates are assumed to be distortion corrected.
 */
class NIFTKOPENCV_EXPORT ProjectCameraRays : public itk::Object
{

public:

  mitkClassMacroItkParent(ProjectCameraRays, itk::Object)
  itkNewMacro(ProjectCameraRays)

  itkSetMacro (ScreenWidth, int);
  itkSetMacro (ScreenHeight, int);
  itkSetMacro (IntrinsicFileName, std::string);
  itkSetMacro (UndistortBeforeProjection, bool);
  itkSetMacro (LensToWorldFileName, std::string);

  std::vector< std::pair < cv::Point3d, cv::Point3d  > > GetRays ();

  void LoadScreenPointsFromFile ( std::string fileName );

  /**
   * \brief writes rays to file. Throws an exception if file can't be written. If fileName is empty writes to std::out
   */
  void WriteOutput ( std::string fileName );

  bool Project();

protected:

  ProjectCameraRays();
  virtual ~ProjectCameraRays();

  ProjectCameraRays(const ProjectCameraRays&); // Purposefully not implemented.
  ProjectCameraRays& operator=(const ProjectCameraRays&); // Purposefully not implemented.

private:

  int         m_ScreenWidth;  //the width of the screen to project
  int         m_ScreenHeight; //the height of the screen to project
  std::string m_LensToWorldFileName; // the transform to put the projected rays in world coordinates

  double      m_RayLength; // the length of a ray

  std::string m_IntrinsicFileName; // the left camera intrinsic parameters

  bool m_UndistortBeforeProjection; //optionally undistort point pairs before triangulation

  std::vector< cv::Point2d > m_ScreenPoints; //a container for the pixel coordinates
  std::vector< std::pair < cv::Point3d, cv::Point3d  > > m_Rays;

  void InitScreenPointsVector (); //Initialise the screen points vector (does nothing if vector already set)

}; // end class

} // end namespace

#endif
