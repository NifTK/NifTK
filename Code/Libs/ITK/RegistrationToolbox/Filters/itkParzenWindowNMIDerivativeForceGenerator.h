/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkParzenWindowNMIDerivativeForceGenerator_h
#define __itkParzenWindowNMIDerivativeForceGenerator_h

#include "itkRegistrationForceFilter.h"
#include "itkLinearlyInterpolatedDerivativeFilter.h"

namespace itk {
/** 
 * \class ParzenWindowNMIDerivativeForceGenerator
 * \brief This class takes as input 2 input images, and outputs 
 * the registration force using Marc's Parzen window approach.
 * (reference to follow).
 * 
 * As of 20090126, this is a bit of a simple implementation, 
 * as we assume that the image has the same number of intensity values
 * as the histogram has bins. So you MUST rescale your image image to
 * fit the histogram first.
 * 
 * \sa RegistrationForceFilter NMILocalHistogramDerivativeForceFilter.
 */

template< class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
class ITK_EXPORT ParzenWindowNMIDerivativeForceGenerator :
  public RegistrationForceFilter<TFixedImage, TMovingImage, TDeformationScalar>
{
public:

  /** Standard "Self" typedef. */
  typedef ParzenWindowNMIDerivativeForceGenerator                         Self;
  typedef RegistrationForceFilter<TFixedImage, TMovingImage, TDeformationScalar> Superclass;
  typedef SmartPointer<Self>                                              Pointer;
  typedef SmartPointer<const Self>                                        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(ParzenWindowNMIDerivativeForceGenerator, RegistrationForceFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);

  /** Standard typedefs. */
  typedef typename Superclass::OutputDataType                 OutputDataType;
  typedef typename Superclass::OutputPixelType                OutputPixelType;
  typedef typename Superclass::OutputImageType                OutputImageType;
  typedef typename OutputImageType::SpacingType               OutputImageSpacingType;
  typedef typename Superclass::InputImageType                 InputImageType;
  typedef typename InputImageType::PixelType                  InputPixelType;
  typedef typename Superclass::InputImageRegionType           RegionType;
  typedef typename Superclass::MetricType                     MetricType;
  typedef typename Superclass::MetricPointer                  MetricPointer;
  typedef typename Superclass::HistogramType                  HistogramType;
  typedef typename Superclass::HistogramPointer               HistogramPointer;
  typedef typename Superclass::HistogramSizeType              HistogramSizeType;
  typedef typename Superclass::HistogramMeasurementVectorType HistogramMeasurementVectorType;
  typedef typename Superclass::HistogramFrequencyType         HistogramFrequencyType;
  typedef typename Superclass::HistogramIteratorType          HistogramIteratorType;
  typedef typename HistogramType::ConstPointer                HistogramConstPointer;
  typedef typename Superclass::MeasureType                    MeasureType;
  typedef LinearlyInterpolatedDerivativeFilter<TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
    ScalarImageGradientFilterType;
  typedef typename ScalarImageGradientFilterType::Pointer 
    ScalarImageGradientFilterPointer;

  /** Connect the ScalarImageGradientFilter. */
  itkSetObjectMacro( ScalarImageGradientFilter, ScalarImageGradientFilterType );

  /** Get a pointer to the ScalarImageGradientFilter.  */
  itkGetConstObjectMacro( ScalarImageGradientFilter, ScalarImageGradientFilterType );

protected:
  
  ParzenWindowNMIDerivativeForceGenerator();
  ~ParzenWindowNMIDerivativeForceGenerator() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Calculate gradient of input image (transformed moving image). */
  ScalarImageGradientFilterPointer m_ScalarImageGradientFilter;

  /** This gets called before ThreadedGenerateData. */
  virtual void BeforeThreadedGenerateData();
  
  /** The "In The Money" method. */
  virtual void ThreadedGenerateData( const RegionType &outputRegionForThread, int);
  
private:

  /**
   * Prohibited copy and assingment. 
   */
  ParzenWindowNMIDerivativeForceGenerator(const Self&); 
  void operator=(const Self&); 

};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkParzenWindowNMIDerivativeForceGenerator.txx"
#endif

#endif
