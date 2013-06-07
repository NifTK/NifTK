/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkSurfaceBasedRegistration_h
#define mitkSurfaceBasedRegistration_h

#include "niftkIGIExports.h"
#include <mitkDataStorage.h>
#include <vtkMatrix4x4.h>
#include <mitkDataNode.h>
#include <mitkSurface.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace mitk {

/**
 * \class SurfaceBasedRegistration
 * \brief Class to perform a surface based registration of two MITK Surfaces/PointSets.
 */
class NIFTKIGI_EXPORT SurfaceBasedRegistration : public itk::Object
{
public:

  mitkClassMacro(SurfaceBasedRegistration, itk::Object);
  itkNewMacro(SurfaceBasedRegistration);

  /**
   * \brief Write My Documentation
   */
  void Update(const mitk::Surface::Pointer fixedNode,
           const mitk::Surface::Pointer movingNode,
           vtkMatrix4x4* transformMovingToFixed);

  enum Method 
  {
    VTK_ICP, //VTK's ICP algorithm, point to surface
    NIFTYSIM //A hypothetical non rigid point to surface algorithm
  };

  itkSetMacro (MaximumIterations, int);
  itkSetMacro (MaximumNumberOfLandmarkPointsToUse, int);
  itkSetMacro (Method, Method);

protected:

  SurfaceBasedRegistration(); // Purposefully hidden.
  virtual ~SurfaceBasedRegistration(); // Purposefully hidden.

  SurfaceBasedRegistration(const SurfaceBasedRegistration&); // Purposefully not implemented.
  SurfaceBasedRegistration& operator=(const SurfaceBasedRegistration&); // Purposefully not implemented.

private:

  int m_MaximumIterations;
  int m_MaximumNumberOfLandmarkPointsToUse;
  Method m_Method;

}; // end class

} // end namespace

#endif
