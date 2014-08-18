/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramFatEstimationFitMetric_h
#define __itkMammogramFatEstimationFitMetric_h

#include <itkSingleValuedCostFunction.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>

namespace itk {
  
/** \class MammogramFatEstimationFitMetric
 * \brief An abstract metric to compute the fit of a model of mammographic fat
 *
 * \section itkMammogramFatEstimationFitMetricCaveats Caveats
 * \li None
 */

class  ITK_EXPORT MammogramFatEstimationFitMetric :
  public SingleValuedCostFunction
{
//  Software Guide : EndCodeSnippet
public:
  typedef MammogramFatEstimationFitMetric  Self;
  typedef SingleValuedCostFunction   Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( MammogramFatEstimationFitMetric, SingleValuedCostFunction );

  /** Method for creation through the object factory. */
  itkNewMacro(Self);


  typedef Superclass::ParametersType ParametersType;
  typedef Superclass::DerivativeType DerivativeType;
  typedef Superclass::MeasureType    MeasureType;

  itkStaticConstMacro( ParametricSpaceDimension, unsigned int, 7 );

  unsigned int GetNumberOfParameters(void) const  
  {
    return ParametricSpaceDimension;
  }

  void GetDerivative( const ParametersType &parameters, 
                      DerivativeType &Derivative ) const
  {
    return;
  }

  virtual MeasureType GetValue( const ParametersType &parameters ) const;

  void GetValueAndDerivative( const ParametersType &parameters,
                              MeasureType &Value, 
                              DerivativeType &Derivative ) const
  {
    Value = this->GetValue( parameters );
    this->GetDerivative( parameters, Derivative );
  }

  virtual void WriteIntensityVsEdgeDistToFile( std::string fileOutputIntensityVsEdgeDist );
  virtual void WriteFitToFile( std::string fileOutputFit,
                               const ParametersType &parameters );


protected:

  MammogramFatEstimationFitMetric();
  virtual ~MammogramFatEstimationFitMetric();
  MammogramFatEstimationFitMetric(const Self &) {}
  void operator=(const Self &) {}
  void PrintSelf(std::ostream & os, Indent indent) const;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammogramFatEstimationFitMetric.txx"
#endif

#endif
