/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : leung@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkLocalHistogramDerivativeForceFilter_h
#define __itkLocalHistogramDerivativeForceFilter_h

#include "itkRegistrationForceFilter.h"

namespace itk {
/** 
 * \class LocalHistogramDerivativeFilter
 * \brief This class takes as input 2 input images, and outputs 
 * the registration force using Bill Crum's local histogram derivative method
 * explained in "Information Theoretic Similarity Measures In Non-Rigid
 * Registration", Crum et al. IPMI 1993.
 * 
 * This abstract base class implements Template Method pattern, so you are expected
 * to subclass it, and implement the ComputeForcePerVoxel according to your 
 * similarity measure.
 * 
 * \sa RegistrationForceFilter NMILocalHistogramDerivativeForceFilter.
 */

template< class TFixedImage, class TMovingImage, class TScalar >
class ITK_EXPORT LocalHistogramDerivativeForceFilter :
  public RegistrationForceFilter<TFixedImage, TMovingImage, TScalar>
{
public:

  /** Standard "Self" typedef. */
  typedef LocalHistogramDerivativeForceFilter                Self;
  typedef RegistrationForceFilter<TFixedImage, TMovingImage, TScalar> Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef SmartPointer<const Self>                           ConstPointer;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(LocalHistogramDerivativeForceFilter, RegistrationForceFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);

  /** Standard typedefs. */
  typedef typename Superclass::OutputPixelType                OutputPixelType;
  typedef typename Superclass::OutputImageType                OutputImageType;
  typedef typename OutputImageType::SpacingType               OutputImageSpacingType;
  typedef typename Superclass::InputImageType                 InputImageType;
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
  
protected:
  
  LocalHistogramDerivativeForceFilter();
  ~LocalHistogramDerivativeForceFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** The "In The Money" method. */
  virtual void ThreadedGenerateData( const RegionType &outputRegionForThread, int);

  /**
   * Override this to supply the algorithm to compute the registration force per voxel. 
   *  
   * \param double totalFrequency Total frequency in the histogram.  
   * \param double jointEntropy Joint entropy H_AB.  
   * \param double fixedImageEntropy Marginal entropy of the fixed image H_A.
   * \param double movingImageEntropy Marginal entropy of the moving image H_B.
   * \param double transformedMovingImageMinusHistogramIndexJointFrequency Joint frequency of the histogram bin in the minus position Freq_mr
   * \param double transformedMovingImagePlusHistogramIndexJointFrequency Joint frequency of the histogram bin in the positive position Freq_mt
   * \param double transformedMovingImageMinusHistogramIndexFrequency Frequency of the histogram bin in the minus position Freq_r.
   * \param double transformedMovingImagePlusHistogramIndexFrequency Frequency of the histogram bin in the positive position Freq_t.
   * \return the registration force using with your algorithm.   
   */
  virtual MeasureType ComputeForcePerVoxel(double totalFrequency, 
                                           double jointEntropy, 
                                           double fixedImageEntropy, 
                                           double movingImageEntropy, 
                                           double transformedMovingImageMinusHistogramIndexJointFrequency,
                                           double transformedMovingImagePlusHistogramIndexJointFrequency,
                                           double transformedMovingImageMinusHistogramIndexFrequency,
                                           double transformedMovingImagePlusHistogramIndexFrequency) const = 0;

private:

  /**
   * Prohibited copy and assingment. 
   */
  LocalHistogramDerivativeForceFilter(const Self&); 
  void operator=(const Self&); 

};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLocalHistogramDerivativeForceFilter.txx"
#endif

#endif
