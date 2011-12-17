/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKSegmentationReliabilityCALCULATOR_H_
#define ITKSegmentationReliabilityCALCULATOR_H_

#include "itkImage.h"
#include "itkObject.h"
#include "itkMacro.h"

namespace itk
{

/**
 * \class SegmentationReliabilityCalculator
 * Calculate the segmentation reliability. 
 *  
 */
template <class TInputImage, class TInputMask, class TOutputImage>
class ITK_EXPORT SegmentationReliabilityCalculator: public Object
{
public:
  /**
   * House keeping for the object factory. 
   */ 
  typedef SegmentationReliabilityCalculator Self;
  typedef Object Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  /**
   * Typedefs. 
   */ 
  typedef typename TInputMask::Pointer TInputMaskPointer;
  typedef typename TInputImage::Pointer TInputImagePointer;
  /** 
   * Method for creation through the object factory. 
   */
  itkNewMacro(Self);  
  /** 
   * Runtime information support. 
   */
  itkTypeMacro(SegmentationReliabilityCalculator, Object);
  /**
   * Macros for setting the input images and masks. 
   */
  itkSetMacro(BaselineImage, TInputImagePointer);
  itkSetMacro(RepeatImage, TInputImagePointer);
  itkGetMacro(SegmentationReliability, double);
  itkSetMacro(BaselineMask, TInputMaskPointer);
  itkSetMacro(RepeatMask, TInputMaskPointer);
  itkSetMacro(NumberOfErosion, unsigned int);
  itkSetMacro(NumberOfDilation, unsigned int);
  itkSetMacro(PaddingValue, typename TInputMask::PixelType);
  itkGetMacro(BSIMask, TInputMaskPointer);
  /**
   * Compute the reliability.
   */
  virtual void Compute(void);
  void PrintSelf(std::ostream& os, Indent indent) const;

protected:
  SegmentationReliabilityCalculator();
  virtual ~SegmentationReliabilityCalculator();
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
  double m_SegmentationReliability;
  /**
   * The input baseline mask for BSI calculation. 
   */
  TInputMaskPointer m_BaselineMask;
  /**
   * The input repeat mask for BSI calculation. 
   */
  TInputMaskPointer m_RepeatMask;
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
   * The input baseline image for BSI calculation.  
   */
  TInputImagePointer m_BaselineImage;
  /**
   * The input repeat image for BSI calculation.  
   */
  TInputImagePointer m_RepeatImage;
  
  
private:  
  SegmentationReliabilityCalculator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSegmentationReliabilityCalculator.txx"
#endif

#endif 


