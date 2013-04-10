/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKDBCImageFilter_H_
#define ITKDBCImageFilter_H_

#include "itkImageToImageFilter.h"
#include <vector>

namespace itk 
{
/**
 * \class DBCImageFilter 
 * Differential-bias-correct a bunch of images. 
 */
template <class TImageType, class TMaskType>
class ITK_EXPORT DBCImageFilter : 
  public ImageToImageFilter<TImageType, TImageType>
{
public:
  /**
   * Basic house keeping. 
   */
  typedef DBCImageFilter Self;
  typedef ImageToImageFilter<TImageType,TImageType> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  itkNewMacro(Self);  
  itkTypeMacro(DBCImageFilter, ImageToImageFilter);
  /**
   * Add input image and mask to the filter. 
   */
  void AddImage(typename TImageType::ConstPointer inputImage, typename TMaskType::ConstPointer inputMask)
  {
    this->m_InputImages.push_back(inputImage); 
    this->m_InputImageMasks.push_back(inputMask); 
  }
  /**
   * Clear all the input images and masks. 
   */
  void ClearImage()
  {
    this->m_InputImages.clear(); 
    this->m_InputImageMasks.clear(); 
  }
  /**
   * Calculate the bias fields. 
   */
  void CalculateBiasFields(); 
  /**
   * Apply calculated bias fields. 
   */
  void ApplyBiasFields(); 
  /**
   * Get the output image. 
   */
  TImageType* GetOutputImage(unsigned int i)
  {
    return this->m_OutputImages[i]; 
  }

protected:
  /**
   * Constructor. 
   */
  DBCImageFilter();
  /**
   * Destructor. 
   */
  virtual ~DBCImageFilter() {}
  /**
   * Erode the image multiple times. 
   */
  void GenerateData();
  /**
   * Quick check on the input images and masks. 
   */
  void InputSanityCheck()
  {
    if (this->m_InputImages.size() != this->m_InputImageMasks.size())
    {
      itkExceptionMacro("Errr... you must supply the same number of images and masks."); 
    }
    if (this->m_InputImages.size() < 2)
    {
      itkExceptionMacro("Errr... you must supply at least two images. Hello? We are doing DBC."); 
    }
  }
  /**
   * Input iamges. 
   */
  std::vector<typename TImageType::ConstPointer> m_InputImages; 
  /**
   * Input iamge masks. 
   */
  std::vector<typename TMaskType::ConstPointer> m_InputImageMasks; 
  /**
   * The final differential bias fields for the input iamges. 
   */
  std::vector<typename TImageType::Pointer> m_BiasFields; 
  /**
   * The bias-corrected images. 
   */
  std::vector<typename TImageType::Pointer> m_OutputImages; 
  /**
   * The amount of expansion needed for the median filter. 
   */
  int m_InputRegionExpansion; 
  /**
   * The radius of the median filter. 
   */
  int m_InputRadius; 
  /**
   * Determine how the differential bias fields of non-consecutive time-points are calculated. 
   * 1: from the images, 2: compose from the differential bias fields of consecutive time-points. 
   */
  int m_InputMode; 
  
private:
  /**
   * Prohibited copy and assingment. 
   */
  DBCImageFilter(const Self&); 
  void operator=(const Self&); 
  
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDBCImageFilter.txx"
#endif




#endif /*ITKDBCImageFilter_H_*/
