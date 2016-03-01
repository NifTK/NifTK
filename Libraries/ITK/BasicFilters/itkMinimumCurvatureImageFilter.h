/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMinimumCurvatureImageFilter_h
#define itkMinimumCurvatureImageFilter_h

#include <itkBinaryFunctorImageFilter.h>
#include <itkNumericTraits.h>

namespace itk
{

/**
 * \class MinimumCurvatureImageFilter
 * \brief Calculates minimum curvature, assuming that the two inputs
 * represent Gaussian Curvature on input 0, and Mean Curvature on input 1.
 *
 * Taken from http://mathworld.wolfram.com/PrincipalCurvatures.html
 *
 * \sa GaussianCurvatureImageFilter
 * \sa MeanCurvatureImageFilter
 */
namespace Functor {

template< class TInput1, class TInput2=TInput1, class TOutput=TInput1>
class MinimumCurvatureFunctor
{
public:
  typedef typename NumericTraits< TInput1 >::AccumulateType AccumulatorType;
  MinimumCurvatureFunctor() {};
  ~MinimumCurvatureFunctor() {};
  bool operator!=( const MinimumCurvatureFunctor & ) const
    {
    return false;
    }
  bool operator==( const MinimumCurvatureFunctor & other ) const
    {
    return !(*this != other);
    }
  inline TOutput operator() ( const TInput1 & K, const TInput2 & H) const
    {
    const AccumulatorType minimumCurvature = H - sqrt(H*H - K);
    return static_cast<TOutput>( minimumCurvature );
    }
};

}
template <class TInputImage1, class TInputImage2=TInputImage1, class TOutputImage=TInputImage1>
class ITK_EXPORT MinimumCurvatureImageFilter :
    public
BinaryFunctorImageFilter<TInputImage1,TInputImage2,TOutputImage,
                         Functor::MinimumCurvatureFunctor<
  typename TInputImage1::PixelType,
  typename TInputImage2::PixelType,
  typename TOutputImage::PixelType>   >


{
public:
  /** Standard class typedefs. */
  typedef MinimumCurvatureImageFilter               Self;
  typedef BinaryFunctorImageFilter<TInputImage1,TInputImage2,TOutputImage,
                                   Functor::MinimumCurvatureFunctor<
    typename TInputImage1::PixelType,
    typename TInputImage2::PixelType,
    typename TOutputImage::PixelType> > Superclass;

  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(MinimumCurvatureImageFilter,
               BinaryFunctorImageFilter);

protected:
  MinimumCurvatureImageFilter() {}
  virtual ~MinimumCurvatureImageFilter() {}

private:
  MinimumCurvatureImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk


#endif
