/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkArunLeastSquaresPointRegistration_h
#define mitkArunLeastSquaresPointRegistration_h

#include <mitkCommon.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <cv.h>

namespace mitk {

/**
 * \class ArunLeastSquaresPointRegistration
 * \brief Performs SVD based registration of two point sets, as in
 *
 * Least-Squares Fitting of two, 3-D Point Sets, Arun, 1987,
 * 10.1109/TPAMI.1987.4767965
 *
 */
class ArunLeastSquaresPointRegistration : public itk::Object
{

public:

  mitkClassMacro(ArunLeastSquaresPointRegistration, itk::Object);
  itkNewMacro(ArunLeastSquaresPointRegistration);

  /**
   * \brief The main method for the calculation, outputs a matrix, and FRE and returns true if calculation deemed OK, and false if failed.
   */
  bool Update(const std::vector<cv::Point3d>& fixedPoints,
              const std::vector<cv::Point3d>& movingPoints,
              cv::Matx44d& outputMatrix,
              double &fiducialRegistrationError);

protected:

  ArunLeastSquaresPointRegistration();
  virtual ~ArunLeastSquaresPointRegistration();

  ArunLeastSquaresPointRegistration(const ArunLeastSquaresPointRegistration&); // Purposefully not implemented.
  ArunLeastSquaresPointRegistration& operator=(const ArunLeastSquaresPointRegistration&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
