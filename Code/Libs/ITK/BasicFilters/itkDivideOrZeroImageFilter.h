/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $
 Revision          : $Revision: 3647 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkDivideOrZeroImageFilter_h
#define __itkDivideOrZeroImageFilter_h

#include "itkBinaryFunctorImageFilter.h"
#include "itkNumericTraits.h"

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
