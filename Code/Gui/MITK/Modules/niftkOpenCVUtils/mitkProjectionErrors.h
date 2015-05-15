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

  mitkClassMacro (ProjectionErrorCalculator, itk::Object);
  itkNewMacro(ProjectionErrorCalculator);

protected:
  ProjectionErrorCalculator();
  virtual ~ProjectionErrorCalculator();

  ProjectionErrorCalculator (const ProjectionErrorCalculator&); // Purposefully not implemented.
  ProjectionErrorCalculator& operator=(const ProjectionErrorCalculator&); // Purposefully not implemented.

};

} // end namespace

#endif



