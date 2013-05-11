/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKEXCLUDEIMAGEFILTER_H_
#define ITKEXCLUDEIMAGEFILTER_H_
#include <itkBinaryFunctorImageFilter.h>
#include <itkNumericTraits.h>


namespace itk
{

/**
 * \class ExcludeImageFilter
 * \brief Performs the connection breaker algorithm as in
 * "Interactive Algorithms for the segmentation and quantification of 3-D MRI scans"
 * Freeborough et. al. CMPB 53 (1997) 15-25.
 * \deprecated Although still a valid function, this class was merged into
 * itk::MIDASMaskByRegionImageFilter and so is currently not used.
 *
 */
namespace Functor
{

template< class TInput1, class TInput2=TInput1, class TOutput=TInput1>
class ConnectionBreak
{
public:
  typedef typename NumericTraits< TInput1 >::AccumulateType AccumulatorType;
  ConnectionBreak() {};
  ~ConnectionBreak() {};

  bool operator!=( const ConnectionBreak & ) const
  {
    return false;
  }

  bool operator==( const ConnectionBreak & other ) const
  {
    return !(*this != other);
  }

  inline TOutput operator() ( const TInput1 & A, const TInput2 & B) const
  {
    //const AccumulatorType sum = A;

    if( (A != 0) && (B == 0))
    {
      return static_cast<TOutput>(1);
    }
    else
    {
      return static_cast<TOutput>(0);
    }
  }

};  //end of class ConnectionBreak

}  //end of namespace Functor


template <class TInputImage1, class TInputImage2=TInputImage1, class TOutputImage=TInputImage1>
class ITK_EXPORT ExcludeImageFilter : public BinaryFunctorImageFilter<TInputImage1,
                                                                      TInputImage2,
                                                                      TOutputImage,
         Functor::ConnectionBreak< typename TInputImage1::PixelType, typename TInputImage2::PixelType,
                                   typename TOutputImage::PixelType > >
{
public:
  /** Standard class typedefs. */
  typedef ExcludeImageFilter               Self;
  typedef BinaryFunctorImageFilter<TInputImage1,TInputImage2,TOutputImage,
                   Functor::ConnectionBreak< typename TInputImage1::PixelType,
                                             typename TInputImage2::PixelType,
                                             typename TOutputImage::PixelType > > Superclass;
  typedef SmartPointer<Self>               Pointer;
  typedef SmartPointer<const Self>         ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ExcludeImageFilter,
               BinaryFunctorImageFilter);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(Input1Input2OutputAdditiveOperatorsCheck,
    (Concept::AdditiveOperators<typename TInputImage1::PixelType,
                                typename TInputImage2::PixelType,
                                typename TOutputImage::PixelType>));
  /** End concept checking */
#endif

protected:
  ExcludeImageFilter() {}
  virtual ~ExcludeImageFilter() {}

private:
  ExcludeImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);     //purposely not implemented

};

} // end namespace itk



#endif /* ITKEXCLUDEIMAGEFILTER_H_ */
