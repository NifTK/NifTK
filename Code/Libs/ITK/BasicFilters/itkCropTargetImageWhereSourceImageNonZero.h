/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCropTargetImageWhereSourceImageNonZero_h
#define __itkCropTargetImageWhereSourceImageNonZero_h

#include <itkBinaryFunctorImageFilter.h>
#include <itkNumericTraits.h>

namespace itk
{

namespace Functor
{

/**
 * \class BWhereANonZeroOtherwiseZero
 * \brief Function object that given two inputs A and B, will return B if A is not zero, and otherwise return 0.
 */
template< class TInput1, class TInput2=TInput1, class TOutput=TInput1 >
class BWhereANonZeroOtherwiseZero
{
public:
  BWhereANonZeroOtherwiseZero() {};
  ~BWhereANonZeroOtherwiseZero() {};
  bool operator!=( const BWhereANonZeroOtherwiseZero& ) const
  {
    return false;
  }
  bool operator==( const BWhereANonZeroOtherwiseZero& other ) const
  {
    return !(*this != other);
  }
  inline TOutput operator()( const TInput1 & A, const TInput2 & B)
  {
    return static_cast<TOutput>( A != 0 ? B : 0 );
  }
}; 

} // end namespace Functor

/**
 * \class CropTargetImageWhereSourceImageNonZeroImageFilter
 * \brief Crops the target (second input) where the source image (first input) is non zero.
 */
template <class TInputImage1, class TInputImage2=TInputImage1, class TOutputImage=TInputImage1>
class ITK_EXPORT CropTargetImageWhereSourceImageNonZeroImageFilter :
    public
BinaryFunctorImageFilter<TInputImage1,TInputImage2,TOutputImage, 
                      Functor::BWhereANonZeroOtherwiseZero<typename TInputImage1::PixelType,
                                                           typename TInputImage2::PixelType,
                                                           typename TOutputImage::PixelType
                                                           >   
                        >
{
public:
  
  /** Standard class typedefs. */
  typedef CropTargetImageWhereSourceImageNonZeroImageFilter        Self;
  typedef BinaryFunctorImageFilter<TInputImage1,TInputImage2,TOutputImage, 
                               Functor::BWhereANonZeroOtherwiseZero< 
                                                                 typename TInputImage1::PixelType, 
                                                                 typename TInputImage2::PixelType,
                                                                 typename TOutputImage::PixelType>   
                                                                >  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CropTargetImageWhereSourceImageNonZeroImageFilter, 
               BinaryFunctorImageFilter);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
//  itkConceptMacro(Input1Input2OutputLogicalOperatorsCheck,
//    (Concept::LogicalOperators<typename TInputImage1::PixelType,
//                               typename TInputImage2::PixelType,
//                               typename TOutputImage::PixelType>));
  /** End concept checking */
#endif

protected:
  CropTargetImageWhereSourceImageNonZeroImageFilter() {}
  virtual ~CropTargetImageWhereSourceImageNonZeroImageFilter() {}

private:
  CropTargetImageWhereSourceImageNonZeroImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace


#endif
