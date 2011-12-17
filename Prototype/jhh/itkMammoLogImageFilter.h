/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMammoLogImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2010-11-18 15:32:39 +0000 (Thu, 18 Nov 2010) $
  Version:   $Revision: 4194 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkMammoLogImageFilter_h
#define __itkMammoLogImageFilter_h

#include "itkUnaryFunctorImageFilter.h"
#include "vnl/vnl_math.h"

namespace itk
{
  
/** \class MammoLogImageFilter
 * \brief Computes the vcl_log(x) pixel-wise
 *
 * \ingroup IntensityImageFilters  Multithreaded
 */
namespace Function {  
  
template< class TInput, class TOutput>
class Log
{
public:
  Log() {}
  ~Log() {}
  bool operator!=( const Log & ) const
    {
    return false;
    }
  bool operator==( const Log & other ) const
    {
    return !(*this != other);
    }
  inline TOutput operator()( const TInput & A ) const
    {
      if (A != 0)
	return static_cast<TOutput>( vcl_log( static_cast<double>( A ) ) );
      else
	return static_cast<TOutput>( 0 );
    }
}; 
}
template <class TInputImage, class TOutputImage>
class ITK_EXPORT MammoLogImageFilter :
    public
UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                        Function::Log< typename TInputImage::PixelType, 
                                       typename TOutputImage::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef MammoLogImageFilter                               Self;
  typedef UnaryFunctorImageFilter<
    TInputImage,TOutputImage, 
    Function::Log< typename TInputImage::PixelType, 
                   typename TOutputImage::PixelType> > Superclass;
  typedef SmartPointer<Self>                           Pointer;
  typedef SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(MammoLogImageFilter, 
               UnaryFunctorImageFilter);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputConvertibleToDoubleCheck,
    (Concept::Convertible<typename TInputImage::PixelType, double>));
  itkConceptMacro(DoubleConvertibleToOutputCheck,
    (Concept::Convertible<double, typename TOutputImage::PixelType>));
  /** End concept checking */
#endif

protected:
  MammoLogImageFilter() {}
  virtual ~MammoLogImageFilter() {}

private:
  MammoLogImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk


#endif
