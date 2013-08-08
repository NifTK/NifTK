/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkUltrasoundPinCalibrationCostFunction_h
#define itkUltrasoundPinCalibrationCostFunction_h

#include <itkMultipleValuedCostFunction.h>
#include <cv.h>

namespace itk {

/**
 * \class UltrasoundPinCalibrationCostFunction
 * \brief Multi-valued cost function adaptor, to plug into Levenberg-Marquardt.
 */
class UltrasoundPinCalibrationCostFunction : public itk::MultipleValuedCostFunction
{

public:

  typedef UltrasoundPinCalibrationCostFunction Self;
  typedef itk::MultipleValuedCostFunction      Superclass;
  typedef itk::SmartPointer<Self>              Pointer;
  typedef itk::SmartPointer<const Self>        ConstPointer;

  itkNewMacro( Self );

  typedef Superclass::ParametersType           ParametersType;
  typedef Superclass::DerivativeType           DerivativeType;
  typedef Superclass::MeasureType              MeasureType;

  void SetMatrices(const std::vector< cv::Mat >& matrices);
  void SetPoints(const std::vector< cv::Point2d > points);

  virtual unsigned int GetNumberOfParameters() const;
  virtual unsigned int GetNumberOfValues(void) const;
  virtual MeasureType GetValue( const ParametersType & parameters ) const;
  virtual void GetDerivative( const ParametersType & parameters, DerivativeType  & derivative ) const;

protected:

  UltrasoundPinCalibrationCostFunction();
  virtual ~UltrasoundPinCalibrationCostFunction();

  UltrasoundPinCalibrationCostFunction(const UltrasoundPinCalibrationCostFunction&); // Purposefully not implemented.
  UltrasoundPinCalibrationCostFunction& operator=(const UltrasoundPinCalibrationCostFunction&); // Purposefully not implemented.

private:

  std::vector< cv::Mat > m_Matrices;
  std::vector< cv::Point2d > m_Points;
  unsigned int m_NumberOfParameters;
  unsigned int m_NumberOfValues;
};

} // end namespace

#endif
