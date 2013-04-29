/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSquaredFunctionImageToImageMetric_h
#define __itkSquaredFunctionImageToImageMetric_h

#include "itkImageToImageMetric.h"
#include "itkCovariantVector.h"
#include "itkPoint.h"

namespace itk
{
/** 
 * \class SquaredFunctionImageToImageMetric
 * \brief Dummy similarity measure, to enable testing of optimizers.
 * 
 * \ingroup RegistrationMetrics
 */
template < typename TFixedImage, typename TMovingImage > 
class ITK_EXPORT SquaredFunctionImageToImageMetric : 
    public ImageToImageMetric< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef SquaredFunctionImageToImageMetric               Self;
  typedef ImageToImageMetric<TFixedImage, TMovingImage >  Superclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(SquaredFunctionImageToImageMetric, ImageToImageMetric);

  /** Types transferred from the base class */
  typedef typename Superclass::TransformType              TransformType;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::MeasureType                MeasureType;

  /** Initializes the metric. This is declared virtual in base class. */
  void Initialize() throw (ExceptionObject) {};

  /**  Get the value for single valued optimizers. */
  MeasureType GetValue( const TransformParametersType & parameters ) const
    {
    	MeasureType result = 0;
    	for (unsigned int i = 0; i < parameters.GetSize(); i++)
    	  {
    	  	result += (parameters.GetElement(i) * parameters.GetElement(i));
    	  }
    	return result;
    }

  /** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType & parameters,
                      DerivativeType  & derivative ) const
    {
    	const unsigned int numberOfParameters = parameters.GetSize();
        derivative = DerivativeType( numberOfParameters );
  
    	for (unsigned int i = 0; i < parameters.GetSize(); i++)
    	  {
    	  	derivative[i] = 2.0 * parameters.GetElement(i);
    	  }
    }

  /**  Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType & parameters,
                              MeasureType& value, DerivativeType& derivative ) const
    {
    	value = this->GetValue(parameters);
    	this->GetDerivative(parameters, derivative);
    }
  
protected:
  
  SquaredFunctionImageToImageMetric() {};
  virtual ~SquaredFunctionImageToImageMetric() {};

private:
  
  SquaredFunctionImageToImageMetric(const Self&); // purposefully not implemented
  void operator=(const Self&);    // purposefully not implemented
  
};

} // end namespace itk

#endif



