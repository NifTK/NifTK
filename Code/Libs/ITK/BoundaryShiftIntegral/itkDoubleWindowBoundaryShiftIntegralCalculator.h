/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKDOUBLEWINDOWBOUNDARYSHIFTINTEGRALCALCULATOR_H_
#define ITKDOUBLEWINDOWBOUNDARYSHIFTINTEGRALCALCULATOR_H_

#include "itkImage.h"
#include "itkObject.h"
#include "itkMacro.h"
#include "itkBoundaryShiftIntegralCalculator.h"

namespace itk
{

/**
 * \class DoubleWindowBoundaryShiftIntegralCalculator
 * \brief Calculate the boundary shift integral using double intensity window.
 * 
 * See the following paper for detais:
 * Freeborough PA and Fox NC, The boundary shift integral: an accurate and 
 * robust measure of cerebral volume changes from registered repeat MRI, 
 * IEEE Trans Med Imaging. 1997 Oct;16(5):623-9.
 *  
 */
template <class TInputImage, class TInputMask, class TOutputImage>
class ITK_EXPORT DoubleWindowBoundaryShiftIntegralCalculator: public BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
{
public:
  /**
   * House keeping for the object factory. 
   */ 
  typedef DoubleWindowBoundaryShiftIntegralCalculator Self;
  typedef BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  /**
   * Typedef the TInputImage::Pointer. 
   */ 
  typedef typename TInputImage::Pointer TInputImagePointer;
  typedef typename TInputMask::Pointer TInputMaskPointer;
  typedef float WeightPixelType; 
  typedef Image<WeightPixelType, TInputImage::ImageDimension> WeightImageType; 
  /** 
   * Method for creation through the object factory. 
   */
  itkNewMacro(Self);  
  /** 
   * Runtime information support. 
   */
  itkTypeMacro(DoubleWindowBoundaryShiftIntegralCalculator, BoundaryShiftIntegralCalculator);
  /**
   * Macros for setting the input images and masks. 
   */
  itkSetMacro(SecondUpperCutoffValue, double);
  itkSetMacro(SecondLowerCutoffValue, double);
  itkGetMacro(FirstBoundaryShiftIntegral, double); 
  itkGetMacro(SecondBoundaryShiftIntegral, double);
  itkSetMacro(WeightImage, typename WeightImageType::Pointer); 
  itkSetMacro(MinSecondWindowWidth, double); 
  itkGetMacro(MinSecondWindowWidth, double); 
  itkGetMacro(SecondBSIMap, typename TOutputImage::Pointer); 

protected:
  DoubleWindowBoundaryShiftIntegralCalculator();
  virtual ~DoubleWindowBoundaryShiftIntegralCalculator() { }; 
  /**
   * Compute the BSI value by integrating the over the BSI mask.
   * \throw ExceptionObject if the lower cut off value is not smaller than the upper cutoff value. 
   */
  virtual void IntegrateOverBSIMask(void) throw (ExceptionObject);
  
protected:
  /**
   * Second upper cutoff value for the normalised intensity.
   */
  double m_SecondUpperCutoffValue;
  /**
   * Second lower cutoff value for the normalised intensity.
   */
  double m_SecondLowerCutoffValue;
  /**
   * BSI from the first intensity window. 
   */
  double m_FirstBoundaryShiftIntegral; 
  /**
   * BSI from the second intensity window. 
   */
  double m_SecondBoundaryShiftIntegral; 
  /**
   * Custom weight image. 
   */
  typename WeightImageType::Pointer m_WeightImage; 
  /**
   * The min. width of the second window. 
   * If less than 0, no min. width. 
   */
  double m_MinSecondWindowWidth; 
  /**
   * A map of BSI values for the second intensity window.  
   */
  typename TOutputImage::Pointer m_SecondBSIMap; 
  
private:
  DoubleWindowBoundaryShiftIntegralCalculator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
  
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDoubleWindowBoundaryShiftIntegralCalculator.txx"
#endif

#endif 


