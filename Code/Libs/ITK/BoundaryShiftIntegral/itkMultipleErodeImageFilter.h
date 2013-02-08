/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKMULTIPLEERODEIMAGEFILTER_H_
#define ITKMULTIPLEERODEIMAGEFILTER_H_

#include "itkImageToImageFilter.h"
#include "itkBinaryCrossStructuringElement.h"
#include "itkBinaryErodeImageFilter.h"

namespace itk 
{
/**
 * \class MultipleErodeImageFilter 
 * Erode a image multiple times. 
 */
template <class TImageType>
class ITK_EXPORT MultipleErodeImageFilter : 
  public ImageToImageFilter<TImageType, TImageType>
{
public:
  /**
   * Basic house keeping. 
   */
  typedef MultipleErodeImageFilter Self;
  typedef ImageToImageFilter<TImageType,TImageType> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  itkNewMacro(Self);  
  itkTypeMacro(MultipleErodeImageFilter, ImageToImageFilter);
  /**
   * Get/Set functions.
   */
  itkSetMacro(NumberOfErosions, unsigned int);
  itkGetMacro(NumberOfErosions, unsigned int);

protected:
  /**
   * Constructor. 
   */
  MultipleErodeImageFilter();
  /**
   * Destructor. 
   */
  virtual ~MultipleErodeImageFilter() {}
  /**
   * Erode the image multiple times. 
   */
  void GenerateData();
  /**
   * A cross structuring element is used in the erosion.  
   */
  typedef itk::BinaryCrossStructuringElement<typename TImageType::PixelType, TImageType::ImageDimension> 
    StructuringElementType;
  /**
   * The erode image filter typedef. 
   */
  typedef itk::BinaryErodeImageFilter<TImageType, TImageType, StructuringElementType> 
    ErodeImageFilterType;
  /**
   * The structuring element used in the erosion.  
   */
  StructuringElementType m_StructuringElement;
  /**
   * The erode image filter. 
   */
  typename ErodeImageFilterType::Pointer m_ErodeImageFilter;
  /**
   * Number of erosion. 
   */
  unsigned int m_NumberOfErosions;
  /**
   * The eroded output image.  
   */
  typename TImageType::Pointer m_ErodedImage;
  /**
   * The value in the image to erode. 
   */
  typename TImageType::PixelType m_ErodeValue;
  /**
   * The value in the image to ignore/erode.
   */
  typename TImageType::PixelType m_BackgroundValue;
  
private:
  /**
   * Prohibited copy and assingment. 
   */
  MultipleErodeImageFilter(const Self&); 
  void operator=(const Self&); 
  
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultipleErodeImageFilter.txx"
#endif




#endif /*ITKMULTIPLEERODEIMAGEFILTER_H_*/
