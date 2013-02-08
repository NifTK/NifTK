/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCrossCorrelationDerivativeForceFilter_h
#define __itkCrossCorrelationDerivativeForceFilter_h

#include "itkRegistrationForceFilter.h"

namespace itk 
{
/** 
 * \class CrossCorrelationDerivativeForceFilter
 * \brief This class takes as input 2 input images, and outputs the registration force based on the derivative of cross correlation. 
 *
 * Implements the registration force derived from cross correlation based on the paper:
 * Freeborough and Fox, Modeling brain deformations in Alzheimer's disease by fluid registration of serial 3D MR images. 
 */
template< class TFixedImage, class TMovingImage, class TScalarType >
class ITK_EXPORT CrossCorrelationDerivativeForceFilter :
  public RegistrationForceFilter<TFixedImage, TMovingImage, TScalarType>
{
public:
  /** 
   * Standard "Self" typedef. 
   */
  typedef CrossCorrelationDerivativeForceFilter Self;
  typedef RegistrationForceFilter<TFixedImage, TMovingImage, TScalarType> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  /** 
   * Run-time type information (and related methods). 
   */
  itkTypeMacro(CrossCorrelationDerivativeForceFilter, RegistrationForceFilter);
  /** 
   * Get the number of dimensions we are working in. 
   */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);
  /** 
   * Method for creation through the object factory. 
   */
  itkNewMacro(Self);
  /**
   * Other craps. 
   */
  typedef typename Superclass::OutputPixelType OutputPixelType;
  typedef typename Superclass::OutputImageType OutputImageType;
  typedef typename OutputImageType::SpacingType OutputImageSpacingType;
  typedef typename Superclass::InputImageType InputImageType;
  typedef typename Superclass::InputImageRegionType RegionType;
  typedef typename Superclass::MetricType MetricType;
  typedef typename Superclass::MetricPointer MetricPointer;
  typedef typename Superclass::MeasureType MeasureType;
  
protected:
  /**
   * Constructor. 
   */ 
  CrossCorrelationDerivativeForceFilter()  { }
  /**
   * Destructor. 
   */
  virtual ~CrossCorrelationDerivativeForceFilter() { }
  /**
   * Compute the force. 
   */
  virtual void GenerateData();
  
private:
  /**
   * Prohibited copy and assingment. 
   */
  CrossCorrelationDerivativeForceFilter(const Self&); 
  void operator=(const Self&); 
}; 

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCrossCorrelationDerivativeForceFilter.txx"
#endif

#endif



