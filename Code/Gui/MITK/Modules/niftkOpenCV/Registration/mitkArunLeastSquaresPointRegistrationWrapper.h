/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkArunLeastSquaresPointRegistrationWrapper_h
#define mitkArunLeastSquaresPointRegistrationWrapper_h

#include "niftkOpenCVExports.h"
#include <mitkCommon.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>

namespace mitk {

/**
 * \class ArunLeastSquaresPointRegistrationWrapper
 * \brief Wrapper for ArunLeastSquaresPointRegistration.
 */
class NIFTKOPENCV_EXPORT ArunLeastSquaresPointRegistrationWrapper : public itk::Object
{

public:

  mitkClassMacro(ArunLeastSquaresPointRegistrationWrapper, itk::Object);
  itkNewMacro(ArunLeastSquaresPointRegistrationWrapper);

  /**
   * \brief \see ArunLeastSquaresPointRegistration::Update()
   */
  bool Update(const mitk::PointSet::Pointer& fixedPoints,
              const mitk::PointSet::Pointer& movingPoints,
              vtkMatrix4x4& matrix,
              double& fiducialRegistrationError
             );

protected:

  ArunLeastSquaresPointRegistrationWrapper();
  virtual ~ArunLeastSquaresPointRegistrationWrapper();

  ArunLeastSquaresPointRegistrationWrapper(const ArunLeastSquaresPointRegistrationWrapper&); // Purposefully not implemented.
  ArunLeastSquaresPointRegistrationWrapper& operator=(const ArunLeastSquaresPointRegistrationWrapper&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
