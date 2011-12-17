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

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkSSDRegistrationForceFilter_h
#define __itkSSDRegistrationForceFilter_h
#include "itkHistogramSimilarityMeasure.h"
#include "itkImageToImageFilter.h"
#include "itkVector.h"
#include "itkImage.h"
#include "itkRegistrationForceFilter.h"

namespace itk {
/** 
 * \class SSDRegistrationForceFilter
 * \brief This class takes as input 2 input images, and outputs the registration force.
 *
 * Implements the registration force using the intensity difference as in Christensen, TMI 1996
 */
template< class TFixedImage, class TMovingImage, class TScalarType >
class ITK_EXPORT SSDRegistrationForceFilter :
  public RegistrationForceFilter<TFixedImage, TMovingImage, TScalarType>
{
public:
  /** 
   * Standard "Self" typedef. 
   */
  typedef SSDRegistrationForceFilter Self;
  typedef RegistrationForceFilter<TFixedImage, TMovingImage, TScalarType> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  /** 
   * Run-time type information (and related methods). 
   */
  itkTypeMacro(SSDRegistrationForceFilter, RegistrationForceFilter);
  /** 
   * Get the number of dimensions we are working in. 
   */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);
  /** 
   * Method for creation through the object factory. 
   */
  itkNewMacro(Self);
  
  /** Standard typedefs. */
  typedef typename Superclass::OutputDataType OutputDataType;
  typedef typename Superclass::OutputPixelType OutputPixelType;
  typedef typename Superclass::OutputImageType OutputImageType;
  
  /**
   * Get/Set the smoothing the moving image before taking gradient. 
   */
  itkSetMacro(Smoothing, bool); 
  itkGetMacro(Smoothing, bool); 
  /**
   * Get/Set the intensity normalisation flag. 
   */
  itkSetMacro(IsIntensityNormalised, bool); 
  itkGetMacro(IsIntensityNormalised, bool); 
  
  SSDRegistrationForceFilter() : m_Smoothing(false), m_IsIntensityNormalised(false) { }
  virtual ~SSDRegistrationForceFilter() { }

protected:
  /**
   * Compute the force. 
   */
  virtual void GenerateData();
  
private:
  /**
   * Smoothing the moving image before taking gradient. 
   */
  bool m_Smoothing; 
  /**
   * Flag to tell if normalising the intensity of the two images by subtracting their means or not. 
   */
  bool m_IsIntensityNormalised; 
  
  /**
   * Prohibited copy and assingment. 
   */
  SSDRegistrationForceFilter(const Self&); 
  void operator=(const Self&); 
  
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSSDRegistrationForceFilter.txx"
#endif

#endif
