/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkInvariantPointCalibration_h
#define mitkInvariantPointCalibration_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <cv.h>

namespace mitk {

/**
 * \class InvariantPointCalibration
 * \brief Base class for Ultrasound Pin calibration and Video Hand-Eye calibration.
 */
class NIFTKOPENCV_EXPORT InvariantPointCalibration : public itk::Object
{

public:

  mitkClassMacro(InvariantPointCalibration, itk::Object);

  itkSetMacro(InvariantPoint, mitk::Point3D);
  itkGetMacro(InvariantPoint, mitk::Point3D);

  itkSetMacro(OptimiseInvariantPoint, bool);
  itkGetMacro(OptimiseInvariantPoint, bool);

  itkSetMacro(TimingLag, double);
  itkGetMacro(TimingLag, double);

  itkSetMacro(OptimiseTimingLag, bool);
  itkGetMacro(OptimiseTimingLag, bool);


  /**
   * \brief Loads a 4x4 matrix from file, decomposes to Rodrigues formula and stores 3 rotations and 3 translation parameters.
   */
  void InitialiseInitialGuess(const std::string& fileName);

  /**
   * \brief Decomposes the matrix to Rodrigues formula and stores 3 rotations and 3 translation parameters.
   */
  void SetInitialGuess(const vtkMatrix4x4& matrix);

  /**
   * \brief Derived classes implement the calibration method.
   */
  virtual void Calibrate() = 0;

protected:

  InvariantPointCalibration();
  virtual ~InvariantPointCalibration();

  InvariantPointCalibration(const InvariantPointCalibration&); // Purposefully not implemented.
  InvariantPointCalibration& operator=(const InvariantPointCalibration&); // Purposefully not implemented.

protected:

  std::vector<double>                                       m_InitialGuess;
  mitk::Point3D                                             m_InvariantPoint;
  bool                                                      m_OptimiseInvariantPoint;
  double                                                    m_TimingLag;
  bool                                                      m_OptimiseTimingLag;
  std::vector< std::pair<unsigned long long, cv::Point3d> > m_Points;


private:

}; // end class

} // end namespace

#endif
