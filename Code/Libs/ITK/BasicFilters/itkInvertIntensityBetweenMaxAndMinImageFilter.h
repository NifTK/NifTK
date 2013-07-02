/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkInvertIntensityBetweenMaxAndMinImageFilter.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkInvertIntensityBetweenMaxAndMinImageFilter_h
#define __itkInvertIntensityBetweenMaxAndMinImageFilter_h

#include "itkUnaryFunctorImageFilter.h"

namespace itk
{

namespace Functor {  
 
template< typename TInput, typename  TOutput>
class InvertIntensityBetweenMaxAndMinTransform
{
public:
  typedef typename NumericTraits< TInput >::RealType RealType;
  InvertIntensityBetweenMaxAndMinTransform() {
    m_Maximum = NumericTraits< TInput >::ZeroValue();
    m_Minimum = NumericTraits< TInput >::ZeroValue();
  }
  ~InvertIntensityBetweenMaxAndMinTransform() {}

  void SetMaximum( TOutput max ) { m_Maximum = max; }
  void SetMinimum( TOutput min ) { m_Minimum = min; }

  bool operator!=( const InvertIntensityBetweenMaxAndMinTransform & other ) const
    {
      if( ( m_Maximum != other.m_Maximum ) ||
          ( m_Minimum != other.m_Minimum ) )
      {
      return true;
      }
    return false;
    }

  bool operator==( const InvertIntensityBetweenMaxAndMinTransform & other ) const
    {
    return !(*this != other);
    }

  inline TOutput operator()( const TInput & x ) const
    {
    TOutput  result = static_cast<TOutput>( m_Maximum - x + m_Minimum );
    return result;
    }
private:
  TInput  m_Maximum;
  TInput  m_Minimum;
}; 

}  // end namespace functor


/** \class InvertIntensityBetweenMaxAndMinImageFilter
 * \brief Invert intensity of an image 
 *
 * InvertIntensityBetweenMaxAndMinImageFilter inverts an image's pixel
 * intensities by subtracting the pixel value, I, from the image
 * maximum, Imax, and adding the minimum, Imin i.e. by computing Inew
 * = Imax - I + Imin. The maximum and minimum values are set to the
 * maximum and minimum of the input image. This filter preserves the
 * range of the input image and can be used to invert, for example, a
 * binary image, a distance map, etc.
 *
 * \author John Hipwell, UCL CMIC.
 *
 * \sa InvertIntensityImageFilter IntensityWindowingImageFilter ShiftScaleImageFilter
 * \ingroup IntensityImageFilters  Multithreaded
 *
 */
template <typename  TInputImage, typename  TOutputImage=TInputImage>
class ITK_EXPORT InvertIntensityBetweenMaxAndMinImageFilter :
    public
    UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                            Functor::InvertIntensityBetweenMaxAndMinTransform< 
  typename TInputImage::PixelType, 
  typename TOutputImage::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef InvertIntensityBetweenMaxAndMinImageFilter    Self;
  typedef UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                                  Functor::InvertIntensityBetweenMaxAndMinTransform< 
    typename TInputImage::PixelType, 
    typename TOutputImage::PixelType> > Superclass;
  typedef SmartPointer<Self>            Pointer;
  typedef SmartPointer<const Self>      ConstPointer;

  typedef typename TOutputImage::PixelType                 OutputPixelType;
  typedef typename TInputImage::PixelType                  InputPixelType;
  typedef typename NumericTraits<InputPixelType>::RealType RealType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Runtime information support. */
  itkTypeMacro(InvertIntensityBetweenMaxAndMinImageFilter, 
               UnaryFunctorImageFilter);

  itkGetConstReferenceMacro( Maximum, InputPixelType );
  itkGetConstReferenceMacro( Minimum, InputPixelType );

  /** Print internal ivars */
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  /** Process to execute before entering the multithreaded section */
  void BeforeThreadedGenerateData(void);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputPixelType>));
  /** End concept checking */
#endif

protected:
  InvertIntensityBetweenMaxAndMinImageFilter();
  virtual ~InvertIntensityBetweenMaxAndMinImageFilter() {};

private:
  InvertIntensityBetweenMaxAndMinImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  InputPixelType        m_Maximum;
  InputPixelType        m_Minimum;
};


  
} // end namespace itk
  
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkInvertIntensityBetweenMaxAndMinImageFilter.txx"
#endif
  
#endif
