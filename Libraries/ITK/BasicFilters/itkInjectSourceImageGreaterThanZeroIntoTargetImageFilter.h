/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkInjectSourceImageGreaterThanZeroIntoTargetImageFilter_h
#define itkInjectSourceImageGreaterThanZeroIntoTargetImageFilter_h

#include <itkBinaryFunctorImageFilter.h>
#include <itkNumericTraits.h>

namespace itk
{

namespace Functor
{

/**
 * \class AWhereNoneZeroOtherwiseB
 * \brief Function object that given two inputs A and B, will return A if A is not zero, and otherwise return B.
 */
template< class TInput1, class TInput2=TInput1, class TOutput=TInput1 >
class AWhereNoneZeroOtherwiseB
{
public:
  AWhereNoneZeroOtherwiseB() {};
  ~AWhereNoneZeroOtherwiseB() {};
  bool operator!=( const AWhereNoneZeroOtherwiseB& ) const
  {
    return false;
  }
  bool operator==( const AWhereNoneZeroOtherwiseB& other ) const
  {
    return !(*this != other);
  }
  inline TOutput operator()( const TInput1 & A, const TInput2 & B)
  {
    return static_cast<TOutput>( A != 0 ? A : B );
  }
}; 

} // end namespace Functor

/**
 * \class InjectSourceImageGreaterThanZeroIntoTargetImageFilter
 * \brief If first input is != 0, then the first input is copied to output,
 * otherwise, the second input pixel is copied to output.
 */
template <class TInputImage1, class TInputImage2=TInputImage1, class TOutputImage=TInputImage1>
class ITK_EXPORT InjectSourceImageGreaterThanZeroIntoTargetImageFilter :
    public
BinaryFunctorImageFilter<TInputImage1,TInputImage2,TOutputImage, 
                         Functor::AWhereNoneZeroOtherwiseB<typename TInputImage1::PixelType,
                                                           typename TInputImage2::PixelType,
                                                           typename TOutputImage::PixelType
                                                           >   
                        >
{
public:
  
  /** Standard class typedefs. */
  typedef InjectSourceImageGreaterThanZeroIntoTargetImageFilter        Self;
  typedef BinaryFunctorImageFilter<TInputImage1,TInputImage2,TOutputImage, 
                                   Functor::AWhereNoneZeroOtherwiseB< 
                                                                     typename TInputImage1::PixelType, 
                                                                     typename TInputImage2::PixelType,
                                                                     typename TOutputImage::PixelType>   
                                                                    >  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(InjectSourceImageGreaterThanZeroIntoTargetImageFilter, 
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
  InjectSourceImageGreaterThanZeroIntoTargetImageFilter() {}
  virtual ~InjectSourceImageGreaterThanZeroIntoTargetImageFilter() {}

private:
  InjectSourceImageGreaterThanZeroIntoTargetImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace


#endif
