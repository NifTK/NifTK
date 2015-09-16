/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkProjectionErrors_h
#define mitkProjectionErrors_h

#include "niftkOpenCVUtilsExports.h"
#include <opencv2/opencv.hpp> 
#include <cv.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkTimeStampsContainer.h>
#include <mitkOpenCVPointTypes.h>

/**
 * \file mitkProjectionErrors.h
 * \brief Derived point types to contain data for projection and analysis
 */
namespace mitk {

/**
* \class Methods to calculate projection / reprojection / triangulation errors 
*/
class NIFTKOPENCVUTILS_EXPORT ProjectionErrorCalculator : public itk::Object
{
public:

  mitkClassMacroItkParent (ProjectionErrorCalculator, itk::Object);
  itkNewMacro(ProjectionErrorCalculator);

  mitk::PickedObject CalculateProjectionError (mitk::PickedObject point , std::string channel); // calculates the projection error in pixels for a single point or line

  void SetProjectedPoints ( const std::vector < mitk::PickedObject >& points );
  void SetClassifierProjectedPoints (const  std::vector < mitk::PickedObject >& points );

protected:
  ProjectionErrorCalculator();
  virtual ~ProjectionErrorCalculator();

  ProjectionErrorCalculator (const ProjectionErrorCalculator&); // Purposefully not implemented.
  ProjectionErrorCalculator& operator=(const ProjectionErrorCalculator&); // Purposefully not implemented.

private:
  mitk::PickedObject FindNearestScreenPoint ( mitk::PickedObject, std::string channel, double* minRatio );

  std::vector < mitk::PickedObject > m_ProjectedPoints; //the projected points to measure errors against
  std::vector < mitk::PickedObject > m_ClassifierProjectedPoints; //optionally we can specify classifier points
  double m_AllowablePointMatchingRatio; //used for optional point classifier

};

} // end namespace

#endif



