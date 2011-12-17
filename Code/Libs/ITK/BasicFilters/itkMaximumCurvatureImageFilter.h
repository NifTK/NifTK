/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-27 22:41:58 +0100 (Mon, 27 Sep 2010) $
 Revision          : $Revision: 3951 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkMaximumCurvatureImageFilter_h
#define __itkMaximumCurvatureImageFilter_h

#include "itkBinaryFunctorImageFilter.h"
#include "itkNumericTraits.h"

namespace itk
{

/**
 * \class MaximumCurvatureImageFilter
 * \brief Calculates maximum curvature, assuming that the two inputs
 * represent Gaussian Curvature on input 0, and Mean Curvature on input 1.
 *
 * Taken from http://mathworld.wolfram.com/PrincipalCurvatures.html
 *
 * \sa GaussianCurvatureImageFilter
 * \sa MeanCurvatureImageFilter
 */
namespace Functor {

template< class TInput1, class TInput2=TInput1, class TOutput=TInput1>
class MaximumCurvatureFunctor
{
public:
  typedef typename NumericTraits< TInput1 >::AccumulateType AccumulatorType;
  MaximumCurvatureFunctor() {};
  ~MaximumCurvatureFunctor() {};
  bool operator!=( const MaximumCurvatureFunctor & ) const
    {
    return false;
    }
  bool operator==( const MaximumCurvatureFunctor & other ) const
    {
    return !(*this != other);
    }
  inline TOutput operator() ( const TInput1 & K, const TInput2 & H) const
    {
    const AccumulatorType maximumCurvature = H + sqrt(H*H - K);
    return static_cast<TOutput>( maximumCurvature );
    }
};

}
template <class TInputImage1, class TInputImage2=TInputImage1, class TOutputImage=TInputImage1>
class ITK_EXPORT MaximumCurvatureImageFilter :
    public
BinaryFunctorImageFilter<TInputImage1,TInputImage2,TOutputImage,
                         Functor::MaximumCurvatureFunctor<
  typename TInputImage1::PixelType,
  typename TInputImage2::PixelType,
  typename TOutputImage::PixelType>   >


{
public:
  /** Standard class typedefs. */
  typedef MaximumCurvatureImageFilter               Self;
  typedef BinaryFunctorImageFilter<TInputImage1,TInputImage2,TOutputImage,
                                   Functor::MaximumCurvatureFunctor<
    typename TInputImage1::PixelType,
    typename TInputImage2::PixelType,
    typename TOutputImage::PixelType> > Superclass;

  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(MaximumCurvatureImageFilter,
               BinaryFunctorImageFilter);

protected:
  MaximumCurvatureImageFilter() {}
  virtual ~MaximumCurvatureImageFilter() {}

private:
  MaximumCurvatureImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk


#endif
