/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkLiuLeastSquaresWithNormalsRegistrationWrapper_h
#define mitkLiuLeastSquaresWithNormalsRegistrationWrapper_h

#include "niftkOpenCVExports.h"
#include <mitkCommon.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>

namespace mitk {

/**
 * \class LiuLeastSquaresWithNormalsRegistrationWrapper
 * \brief \see LiuLeastSquaresWithNormalsRegistration
 */
class NIFTKOPENCV_EXPORT LiuLeastSquaresWithNormalsRegistrationWrapper : public itk::Object
{
public:

  mitkClassMacro(LiuLeastSquaresWithNormalsRegistrationWrapper, itk::Object);
  itkNewMacro(LiuLeastSquaresWithNormalsRegistrationWrapper);

  /**
   * \brief The main method for the calculation, outputs a matrix, and FRE and returns true if calculation deemed OK, and false if failed.
   */
  bool Update(const mitk::PointSet::Pointer& fixedPoints,
              const mitk::PointSet::Pointer& fixedNormals,
              const mitk::PointSet::Pointer& movingPoints,
              const mitk::PointSet::Pointer& movingNormals,
              vtkMatrix4x4& matrix,
              double &fiducialRegistrationError);

protected:

  LiuLeastSquaresWithNormalsRegistrationWrapper();
  virtual ~LiuLeastSquaresWithNormalsRegistrationWrapper();

  LiuLeastSquaresWithNormalsRegistrationWrapper(const LiuLeastSquaresWithNormalsRegistrationWrapper&); // Purposefully not implemented.
  LiuLeastSquaresWithNormalsRegistrationWrapper& operator=(const LiuLeastSquaresWithNormalsRegistrationWrapper&); // Purposefully not implemented.
};

} // end namespace

#endif
