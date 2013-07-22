/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkLuiLeastSquaresWithNormalsRegistration_h
#define mitkLuiLeastSquaresWithNormalsRegistration_h

#include <mitkCommon.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <cv.h>

namespace mitk {

/**
 * \class LuiLeastSquaresWithNormalsRegistration
 * \brief Performs SVD based registration of two point sets with surface normals, as in Liu, Fitzpatrick 2003.
 *
 * See also <a href="http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=758228">this link</a>.
 */
class LuiLeastSquaresWithNormalsRegistration : public itk::Object
{

public:

  mitkClassMacro(LuiLeastSquaresWithNormalsRegistration, itk::Object);
  itkNewMacro(LuiLeastSquaresWithNormalsRegistration);

  /**
   * \brief The main method for the calculation, outputs a matrix, and FRE and returns true if calculation deemed OK, and false if failed.
   */
  bool Update(const std::vector<cv::Point3d>& fixedPoints,
              const std::vector<cv::Point3d>& fixedNormals,
              const std::vector<cv::Point3d>& movingPoints,
              const std::vector<cv::Point3d>& movingNormals,
              cv::Matx44d& outputMatrix,
              double &fiducialRegistrationError);

protected:

  LuiLeastSquaresWithNormalsRegistration();
  virtual ~LuiLeastSquaresWithNormalsRegistration();

  LuiLeastSquaresWithNormalsRegistration(const LuiLeastSquaresWithNormalsRegistration&); // Purposefully not implemented.
  LuiLeastSquaresWithNormalsRegistration& operator=(const LuiLeastSquaresWithNormalsRegistration&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
