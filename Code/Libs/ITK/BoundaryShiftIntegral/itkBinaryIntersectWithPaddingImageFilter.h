/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKBINARYINTERSECTWITHPADDINGIMAGE_H_
#define ITKBINARYINTERSECTWITHPADDINGIMAGE_H_

#include "itkBinaryFunctorImageFilter.h"

namespace itk
{
  
namespace Functor {  
  
/**
 * \class BinaryIntersectWithPadding
 * \brief Provide the operator to calculate the intersection of two pixels
 * using padding value: return 1, if both pixels do not equal to the padding value,
 * return 0, otherwise.
 */
template<class TInput, class TOutput>
class BinaryIntersectWithPadding
{
public:
  BinaryIntersectWithPadding() {};
  ~BinaryIntersectWithPadding() {};
  bool operator!=(const BinaryIntersectWithPadding &) const
  {
    return false;
  }
  bool operator==(const BinaryIntersectWithPadding& other) const
  {
    return !(*this != other);
  }
  /**
   * Operator to calculate the intersection of two pixels using padding value.
   * \param const TInput& value1: pixel value 1.
   * \param const TInput& value2: pixel value 2.
   * \return 1, if both pixels do not equal to the padding value. Return 0, otherwise.  
   */
  inline TOutput operator()(const TInput& value1, const TInput& value2)
  {
    TInput returnValue = 0;
    
    if (value1 != m_PaddingValue && value2 != m_PaddingValue)
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
 * \class BinaryIntersectWithPaddingImageFilter
 * \brief Calculate the intersection of two images using padding value.
 *
 * In each pixel location, 
 * set to 1, if both pixels do not equal to the padding value.
 * set to 0, otherwise.
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT BinaryIntersectWithPaddingImageFilter :
    public BinaryFunctorImageFilter<TInputImage,TInputImage,TOutputImage, 
                                    Functor::BinaryIntersectWithPadding< 
                                      typename TInputImage::PixelType, 
                                      typename TOutputImage::PixelType> >
{
public:
  /** 
   * Standard class typedefs. 
   */
  typedef BinaryIntersectWithPaddingImageFilter Self;
  typedef BinaryFunctorImageFilter<TInputImage,TInputImage,TOutputImage, 
                                   Functor::BinaryIntersectWithPadding< 
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
  itkTypeMacro(BinaryIntersectWithPaddingImageFilter, BinaryFunctorImageFilter);
  /**
   * Get/Set functions.
   */
  itkSetMacro(PaddingValue, typename TInputImage::PixelType);
  itkGetMacro(PaddingValue, typename TInputImage::PixelType);

protected:
  BinaryIntersectWithPaddingImageFilter()
  {
    m_PaddingValue = NumericTraits<typename TInputImage::PixelType>::Zero;
  }
  virtual ~BinaryIntersectWithPaddingImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void BeforeThreadedGenerateData();

private:
  BinaryIntersectWithPaddingImageFilter(const Self&); 
  void operator=(const Self&); 
  /**
   * Padding/background value to be ignored in the images. 
   */
  typename TInputImage::PixelType m_PaddingValue;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBinaryIntersectWithPaddingImageFilter.txx"
#endif


#endif


