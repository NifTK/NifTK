/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef INTENSITY_NORMALISATION_CALCULATOR_H_
#define INTENSITY_NORMALISATION_CALCULATOR_H_

#include "itkImage.h"
#include "itkObject.h"
#include "itkMacro.h"

namespace itk
{

/**
 * \class IntensityNormalisationCalculator
 * \brief Calculates the means to normalise the intensities of two scans.
 * Calculate the means used to normalise the intensities of two scans 
 * acquired in different dates, using the following steps:
 * 1. The intersection of the two masks is found.
 * 2. It is eroded once. 
 * 3. Return the mean intensities of the eroded intersect regions in the two images.   
 */
template <class TInputImage, class TInputMask>
class ITK_EXPORT IntensityNormalisationCalculator : public Object
{
public:
  /**
   * House keeping for the object factory. 
   */ 
  typedef IntensityNormalisationCalculator Self;
  typedef Object Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  /** 
   * Method for creation through the object factory. 
   */
  itkNewMacro(Self);  
  /** 
   * Runtime information support. 
   */
  itkTypeMacro(IntensityNormalisationCalculator, Object);
  /**
   * Helper typedefs. 
   */ 
  typedef typename TInputImage::Pointer TInputImagePointer;
  typedef typename TInputMask::Pointer TInputMaskPointer;
  typedef typename TInputMask::PixelType TInputMaskPixelType;
  /**
   * Macros for setting the input images and masks. 
   */
  itkSetMacro(InputImage1, TInputImagePointer); 
  itkSetMacro(InputImage2, TInputImagePointer);
  itkSetMacro(InputMask1, TInputMaskPointer);
  itkSetMacro(InputMask2, TInputMaskPointer);
  itkGetMacro(NormalisationMean1, double);
  itkGetMacro(NormalisationMean2, double);
  itkSetMacro(PaddingValue, TInputMaskPixelType);
  /**
   * Compute the means of the baseline and repeat images for intensity normalisation.
   */
  void Compute();

protected:
  IntensityNormalisationCalculator();
  virtual ~IntensityNormalisationCalculator() {};
  void PrintSelf(std::ostream& os, Indent indent) const;
  
private:
  IntensityNormalisationCalculator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  /**
   * The normalisation mean for image 1. 
   */
  double m_NormalisationMean1;
  /**
   * The normalisation mean for image 2. 
   */
  double m_NormalisationMean2;
  /**
   * The input image 1 for normalisation.  
   */
  TInputImagePointer m_InputImage1;
  /**
   * The input image 2 for normalisation.  
   */
  TInputImagePointer m_InputImage2;
  /**
   * The input mask 1 for normalisation. 
   */
  TInputMaskPointer m_InputMask1;
  /**
   * The input mask 2 for normalisation. 
   */
  TInputMaskPointer m_InputMask2;
  /**
   * Padding/background value to be ignored in the masks. 
   */
  TInputMaskPixelType m_PaddingValue;
  
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkIntensityNormalisationCalculator.txx"
#endif

#endif 


