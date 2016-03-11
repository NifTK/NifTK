/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkNegateImageFilter_h
#define itkNegateImageFilter_h

#include <itkUnaryFunctorImageFilter.h>
#include <itkConceptChecking.h>

namespace itk
{

/** \class NegateImageFilter
 * \brief Computes the -1.0 * x pixel-wise
 *
 * \ingroup IntensityImageFilters  Multithreaded
 */
namespace Functor {

template< typename TInput, typename TOutput>
class Negate
{
public:
  Negate() {}
  ~Negate() {}
  bool operator!=( const Negate & ) const
    {
    return false;
    }
  bool operator==( const Negate & other ) const
    {
    return !(*this != other);
    }
  inline TOutput operator()( const TInput & A ) const
    {
      return -static_cast<TOutput>( A );
    }
};
}

template <typename TInputImage, typename TOutputImage>
class ITK_EXPORT NegateImageFilter :
    public
UnaryFunctorImageFilter<TInputImage,TOutputImage,
      Functor::Negate< 
      typename TInputImage::PixelType,
      typename TOutputImage::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef NegateImageFilter  Self;
  typedef UnaryFunctorImageFilter<TInputImage,TOutputImage,
                                  Functor::Negate< typename TInputImage::PixelType,
                                                 typename TOutputImage::PixelType> >  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(NegateImageFilter,
               UnaryFunctorImageFilter);

protected:
  NegateImageFilter() {}
  virtual ~NegateImageFilter() {}

private:
  NegateImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk


#endif
