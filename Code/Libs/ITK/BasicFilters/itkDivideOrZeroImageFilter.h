/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkDivideOrZeroImageFilter_h
#define itkDivideOrZeroImageFilter_h

#include <itkBinaryFunctorImageFilter.h>
#include <itkNumericTraits.h>

namespace itk
{

/** \class DivideOrZeroImageFilter
 * \brief Implements an operator for pixel-wise division of two images, and where divisor is zero, outputs zero.
 */

namespace Function {

template< class TInput1, class TInput2, class TOutput>
class DivOrZero
{
public:
  DivOrZero() {};
  ~DivOrZero() {};
  bool operator!=( const DivOrZero & ) const
    {
    return false;
    }
  bool operator==( const DivOrZero & other ) const
    {
    return !(*this != other);
    }
  inline TOutput operator()( const TInput1 & A, const TInput2 & B) const
    {
    if(B != (TInput2) 0)
      {
      return (TOutput)(A / B);
      }
    else
      {
      return 0;
      }
    }
};
}

template <class TInputImage1, class TInputImage2, class TOutputImage>
class ITK_EXPORT DivideOrZeroImageFilter :
    public
BinaryFunctorImageFilter<TInputImage1,TInputImage2,TOutputImage,
                         Function::DivOrZero<
  typename TInputImage1::PixelType,
  typename TInputImage2::PixelType,
  typename TOutputImage::PixelType>   >
{
public:
  /**
   * Standard "Self" typedef.
   */
  typedef DivideOrZeroImageFilter  Self;

  /**
   * Standard "Superclass" typedef.
   */
  typedef BinaryFunctorImageFilter<TInputImage1,TInputImage2,TOutputImage,
                                   Function::DivOrZero<
    typename TInputImage1::PixelType,
    typename TInputImage2::PixelType,
    typename TOutputImage::PixelType>
  > Superclass;

  /**
   * Smart pointer typedef support
   */
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /**
   * Method for creation through the object factory.
   */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(DivideOrZeroImageFilter,
               BinaryFunctorImageFilter);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(IntConvertibleToInput2Check,
    (Concept::Convertible<int, typename TInputImage2::PixelType>));
  itkConceptMacro(Input1Input2OutputDivisionOperatorsCheck,
    (Concept::DivisionOperators<typename TInputImage1::PixelType,
                                typename TInputImage2::PixelType,
                                typename TOutputImage::PixelType>));
  /** End concept checking */
#endif

protected:
  DivideOrZeroImageFilter() {}
  virtual ~DivideOrZeroImageFilter() {}
  DivideOrZeroImageFilter(const Self&) {}
  void operator=(const Self&) {}

};

} // end namespace itk


#endif
