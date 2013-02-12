/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKBOUNDARYSHIFTINTEGRALCALCULATOR_H_
#define ITKBOUNDARYSHIFTINTEGRALCALCULATOR_H_

#include "itkImage.h"
#include "itkObject.h"
#include "itkMacro.h"

namespace itk
{

/**
 * \class BoundaryShiftIntegralCalculator
 * \brief Calculate the boundary shift integral.
 * 
 * See the following paper for detais:
 * Freeborough PA and Fox NC, The boundary shift integral: an accurate and 
 * robust measure of cerebral volume changes from registered repeat MRI, 
 * IEEE Trans Med Imaging. 1997 Oct;16(5):623-9.
 *  
 */
template <class TInputImage, class TInputMask, class TOutputImage>
class ITK_EXPORT BoundaryShiftIntegralCalculator: public Object
{
public:
  /**
   * House keeping for the object factory. 
   */ 
  typedef BoundaryShiftIntegralCalculator Self;
  typedef Object Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  /**
   * Typedef the TInputImage::Pointer. 
   */ 
  typedef typename TInputImage::Pointer TInputImagePointer;
  typedef typename TInputMask::Pointer TInputMaskPointer;
  /** 
   * Method for creation through the object factory. 
   */
  itkNewMacro(Self);  
  /** 
   * Runtime information support. 
   */
  itkTypeMacro(BoundaryShiftIntegralCalculator, Object);
  /**
   * Macros for setting the input images and masks. 
   */
  itkGetMacro(BoundaryShiftIntegral, double);
  itkSetMacro(BaselineImage, TInputImagePointer);
  itkSetMacro(BaselineMask, TInputMaskPointer);
  itkSetMacro(RepeatImage, TInputImagePointer);
  itkSetMacro(RepeatMask, TInputMaskPointer);
  itkSetMacro(NumberOfErosion, unsigned int);
  itkSetMacro(NumberOfDilation, unsigned int);
  itkSetMacro(NumberOfSubROIDilation, unsigned int);
  itkSetMacro(SubROIMask, TInputMaskPointer);
  itkSetMacro(UpperCutoffValue, double);
  itkSetMacro(LowerCutoffValue, double);
  itkSetMacro(BaselineIntensityNormalisationFactor, double);
  itkSetMacro(RepeatIntensityNormalisationFactor, double);
  itkSetMacro(PaddingValue, typename TInputMask::PixelType);
  itkGetMacro(BSIMask, TInputMaskPointer);
  itkGetMacro(BSIMap, typename TOutputImage::Pointer); 
  /**
   * Compute the BSI.
   */
  virtual void Compute(void);
  void PrintSelf(std::ostream& os, Indent indent) const;
  /**
  * Calculate the simple linear regression between x and y.  
  */
  static void PerformLinearRegression(const std::vector<double>& x, const std::vector<double>& y, double* slope, double* intercept)
  {
    assert(x.size() == y.size());
    
    double correlation = 0.0;
    double meanX = 0.0;
    double meanY = 0.0;
    double varianceX = 0.0;
    int sizeOfArray = x.size();
    
    for (int arrayIndex = 0; arrayIndex < sizeOfArray; arrayIndex++)
    {
      meanX += x[arrayIndex];
      meanY += y[arrayIndex];
    }
    meanX /= sizeOfArray;
    meanY /= sizeOfArray;
    
    for (int arrayIndex = 0; arrayIndex < sizeOfArray; arrayIndex++)
    {
      correlation += (x[arrayIndex]-meanX)*(y[arrayIndex]-meanY);
      varianceX += (x[arrayIndex]-meanX)*(x[arrayIndex]-meanX);
    }
    *slope = correlation/varianceX;
    *intercept = meanY-(*slope)*meanX;
  }

protected:
  BoundaryShiftIntegralCalculator();
  virtual ~BoundaryShiftIntegralCalculator();
  /**
   * Compute the mask that is used for the BSI integration.
   */
  virtual void ComputeBSIMask(void);
  /**
   * Compute the BSI value by integrating the over the BSI mask.
   * \throw ExceptionObject if the lower cut off vakye is not smaller than the 
   */
  virtual void IntegrateOverBSIMask(void) throw (ExceptionObject);
  
protected:
  /**
   * Compute the eroded intersect mask by 
   * 1. intesect the input baseline and repeat masks.
   * 2. erode the intersection. 
   */
  void ComputeErodedIntersectMask(void);
  /**
   * Compute the dilated union mask by 
   * 1. union the input baseline and repeat masks.
   * 2. dilate the union. 
   */
  void ComputeDilatedUnionMask(void);
  /**
   * The boundary shift integral value. 
   */
  double m_BoundaryShiftIntegral;
  /**
   * The input baseline image for BSI calculation.  
   */
  TInputImagePointer m_BaselineImage;
  /**
   * The input baseline mask for BSI calculation. 
   */
  TInputMaskPointer m_BaselineMask;
  /**
   * The input repeat image for BSI calculation.  
   */
  TInputImagePointer m_RepeatImage;
  /**
   * The input repeat mask for BSI calculation. 
   */
  TInputMaskPointer m_RepeatMask;
  /**
   * The number of dilation applied to the sub ROI.
   */
  unsigned int m_NumberOfSubROIDilation;
  /**
   * Sub ROI used to apply to the boundary BSI mask before cal. 
   */
  TInputMaskPointer m_SubROIMask;
  /**
   * The number of erosion applied to the intersect mask.
   */
  unsigned int m_NumberOfErosion;
  /**
   * The binary eroded intersect mask, with 0 and 1, from the input baseline and repeat masks.
   */
  TInputMaskPointer m_ErodedIntersectMask;
  /**
   * The number of dilation applied to the union mask.
   */
  unsigned int m_NumberOfDilation;
  /**
   * The binary dilated intersect mask, with 0 and 1, from the input baseline and repeat masks.
   */
  TInputMaskPointer m_DilatedUnionMask;
  /**
   * The padding/background value in all the input masks.
   */
  typename TInputMask::PixelType m_PaddingValue;
  /**
   * The BSI mask, with 0 and 1, which the integration will be done.  
   */
  TInputMaskPointer m_BSIMask;
  /**
   * Upper cutoff value for the normalised intensity.
   * Default: 0.25. 
   */
  double m_UpperCutoffValue;
  /**
   * Lower cutoff value for the normalised intensity.
   * Default: 0.75. 
   */
  double m_LowerCutoffValue;
  /**
   * Baseline intensity normalisation factor - the basseline image intensity 
   * will be divided by this factor.  
   */
  double m_BaselineIntensityNormalisationFactor;
  /**
   * Repeat intensity normalisation factor - the repeat image intensity 
   * will be divided by this factor.  
   */
  double m_RepeatIntensityNormalisationFactor;
  /**
   * A map of BSI values. 
   */
  typename TOutputImage::Pointer m_BSIMap; 
  
private:  
  BoundaryShiftIntegralCalculator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBoundaryShiftIntegralCalculator.txx"
#endif

#endif 


