/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKMULTIPLEDILATEIMAGEFILTER_H_
#define ITKMULTIPLEDILATEIMAGEFILTER_H_

#include "itkImageToImageFilter.h"
#include "itkBinaryCrossStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"

namespace itk 
{

/**
 * \class MultipleDilateImageFilter 
 * Dilate a image multiple times. 
 */
template <class TImageType>
class ITK_EXPORT MultipleDilateImageFilter : 
  public ImageToImageFilter<TImageType, TImageType>
{
public:
  /**
   * Basic house keeping. 
   */
  typedef MultipleDilateImageFilter Self;
  typedef ImageToImageFilter<TImageType,TImageType> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  itkNewMacro(Self);  
  itkTypeMacro(MultipleDilateImageFilter, ImageToImageFilter);
  /**
   * Get/Set functions.
   */
  itkSetMacro(NumberOfDilations, unsigned int);
  itkGetMacro(NumberOfDilations, unsigned int);
  itkSetMacro(DilateValue, typename TImageType::PixelType); 
  itkGetMacro(DilateValue, typename TImageType::PixelType); 

protected:
  /**
   * Constructor. 
   */
  MultipleDilateImageFilter();
  /**
   * Destructor. 
   */
  virtual ~MultipleDilateImageFilter() {}
  /**
   * Dilate the image multiple times. 
   */
  void GenerateData();
  /**
   * A cross structuring element is used in the dilation.  
   */
  typedef itk::BinaryCrossStructuringElement<typename TImageType::PixelType, TImageType::ImageDimension> 
    StructuringElementType;
  /**
   * The dilate image filter typedef. 
   */
  typedef itk::BinaryDilateImageFilter<TImageType, TImageType, StructuringElementType> 
    DilateImageFilterType;
  /**
   * The structuring element used in the dilation.  
   */
  StructuringElementType m_StructuringElement;
  /**
   * The dilate image filter. 
   */
  typename DilateImageFilterType::Pointer m_DilateImageFilter;
  /**
   * Number of dilation. 
   */
  unsigned int m_NumberOfDilations;
  /**
   * The dilated output image.  
   */
  typename TImageType::Pointer m_DilatedImage;
  /**
   * The value in the image to dilate. 
   */
  typename TImageType::PixelType m_DilateValue;
  /**
   * The value in the image to ignore/erode.
   */
  typename TImageType::PixelType m_BackgroundValue;
  
private:
  /**
   * Prohibited copy and assingment. 
   */
  MultipleDilateImageFilter(const Self&); 
  void operator=(const Self&); 
  
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultipleDilateImageFilter.txx"
#endif



#endif /*ITKMULTIPLEDILATEIMAGEFILTER_H_*/
