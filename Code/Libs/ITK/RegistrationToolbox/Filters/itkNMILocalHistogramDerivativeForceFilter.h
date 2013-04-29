/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkNMILocalHistogramDerivativeForceFilter_h
#define __itkNMILocalHistogramDerivativeForceFilter_h

#include "itkLocalHistogramDerivativeForceFilter.h"

namespace itk {
/** 
 * \class NMILocalHistogramDerivativeFilter
 * \brief Implements LocalHistogramDerivativeFilter for Normalized Mutual Information.

 * \sa RegistrationForceFilter LocalHistogramDerivativeFilter
 */

template< class TFixedImage, class TMovingImage, class TScalar >
class ITK_EXPORT NMILocalHistogramDerivativeForceFilter :
  public LocalHistogramDerivativeForceFilter<TFixedImage, TMovingImage, TScalar>
{
public:

  /** Standard "Self" typedef. */
  typedef NMILocalHistogramDerivativeForceFilter                         Self;
  typedef LocalHistogramDerivativeForceFilter<TFixedImage, TMovingImage, TScalar> Superclass;
  typedef SmartPointer<Self>                                             Pointer;
  typedef SmartPointer<const Self>                                       ConstPointer;
  typedef typename Superclass::MeasureType                               MeasureType;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(NMILocalHistogramDerivativeForceFilter, LocalHistogramDerivativeForceFilter);

protected:
  
  NMILocalHistogramDerivativeForceFilter() {};
  ~NMILocalHistogramDerivativeForceFilter() {};

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
  MeasureType ComputeForcePerVoxel(double totalFrequency, 
                                   double jointEntropy, 
                                   double fixedImageEntropy, 
                                   double transformedMovingImageEntropy, 
                                   double transformedMovingImageMinusHistogramIndexJointFrequency,
                                   double transformedMovingImagePlusHistogramIndexJointFrequency,
                                   double transformedMovingImageMinusHistogramIndexFrequency,
                                   double transformedMovingImagePlusHistogramIndexFrequency) const
    {
      
      double result = (1.0/(jointEntropy*jointEntropy*totalFrequency))*
               ((fixedImageEntropy+transformedMovingImageEntropy)*vcl_log(transformedMovingImageMinusHistogramIndexJointFrequency/transformedMovingImagePlusHistogramIndexJointFrequency) - 
                (jointEntropy*vcl_log(transformedMovingImageMinusHistogramIndexFrequency/transformedMovingImagePlusHistogramIndexFrequency)));

      /*
      std::cerr << totalFrequency << "," \
                << jointEntropy << "," \
                << fixedImageEntropy << "," \
                << transformedMovingImageEntropy << "," \
                << transformedMovingImageMinusHistogramIndexJointFrequency << "," \
                << transformedMovingImagePlusHistogramIndexJointFrequency << "," \
                << transformedMovingImageMinusHistogramIndexFrequency << "," \
                << transformedMovingImagePlusHistogramIndexFrequency << "," \
                << result << std::endl;
      */
      
      return result;
    }


private:

  /**
   * Prohibited copy and assingment. 
   */
  NMILocalHistogramDerivativeForceFilter(const Self&); 
  void operator=(const Self&); 

};

} // end namespace

#endif
