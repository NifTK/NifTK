/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKBINARISEUSINGPADDINGIMAGEFILTER_H_
#define ITKBINARISEUSINGPADDINGIMAGEFILTER_H_

#include "itkUnaryFunctorImageFilter.h"

namespace itk
{
  
namespace Functor {  
  
/**
 * \class BinariseUsingPadding
 * \brief Provide the operator to binarise a pixel: return 0, if the pixel is equal to the padding value, return 1, otherwise.
 */
template<class TInput, class TOutput>
class BinariseUsingPadding
{
public:
  BinariseUsingPadding() {};
  ~BinariseUsingPadding() {};
  bool operator!=(const BinariseUsingPadding&) const
  {
    return false;
  }
  bool operator==(const BinariseUsingPadding& other) const
  {
    return !(*this != other);
  }
  /**
   * Provide the operator to binarise an pixel. 
   * \param const TInput& value: pixel value.
   * return 0, if the pixel is equal to the padding value.
   * return 1, otherwise.
   */
  inline TOutput operator()(const TInput& value)
  {
    TOutput returnValue = 0;
    
    if (value != m_PaddingValue)
      returnValue = 1;
    return returnValue;
  }
  /**
   * Get/Set functions.
   */
  void SetPaddingValue(const TInput& paddingValue) 
  {
    m_PaddingValue = paddingValue;
  }
  TInput GetPaddingValue() const 
  {
    return m_PaddingValue; 
  }
  
private:
  /**
   * Padding/background value to be ignored in the images. 
   */
  TInput m_PaddingValue;
}; 
} // End of namespace functor.


/**
 * \class BinariseUsingPaddingImageFilter
 * \brief Binarise the image using using padding value.
 *
 * In each pixel location, 
 * set to 1, if pixel is not equal to the padding value.
 * set to 0, otherwise.
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT BinariseUsingPaddingImageFilter :
    public UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                                    Functor::BinariseUsingPadding< 
                                      typename TInputImage::PixelType, 
                                      typename TOutputImage::PixelType> >
{
public:
  /** 
   * Standard class typedefs. 
   */
  typedef BinariseUsingPaddingImageFilter Self;
  typedef UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                                   Functor::BinariseUsingPadding< 
                                     typename TInputImage::PixelType, 
                                     typename TOutputImage::PixelType> >  Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  /** 
   * Method for creation through the object factory. 
   */
  itkNewMacro(Self);
  /** 
   * Runtime information support. 
   */
  itkTypeMacro(BinariseUsingPaddingImageFilter, UnaryFunctorImageFilter);
  /**
   * Get/Set functions.
   */
  itkSetMacro(PaddingValue, typename TInputImage::PixelType);
  itkGetMacro(PaddingValue, typename TInputImage::PixelType);

protected:
  BinariseUsingPaddingImageFilter()
  {
    m_PaddingValue = NumericTraits<typename TInputImage::PixelType>::Zero;
  }
  virtual ~BinariseUsingPaddingImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void BeforeThreadedGenerateData();

private:
  BinariseUsingPaddingImageFilter(const Self&); 
  void operator=(const Self&); 
  /**
   * Padding/background value to be ignored in the images. 
   */
  typename TInputImage::PixelType m_PaddingValue;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBinariseUsingPaddingImageFilter.txx"
#endif


#endif 
