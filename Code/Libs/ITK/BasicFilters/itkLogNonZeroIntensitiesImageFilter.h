/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkLogNonZeroIntensitiesImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2010-11-18 15:32:39 +0000 (Thu, 18 Nov 2010) $
  Version:   $Revision: 4194 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkLogNonZeroIntensitiesImageFilter_h
#define __itkLogNonZeroIntensitiesImageFilter_h

#include <itkUnaryFunctorImageFilter.h>
#include <vnl/vnl_math.h>

namespace itk
{
  
/** \class LogNonZeroIntensitiesImageFilter
 * \brief Computes the vcl_log(x) pixel-wise of non-zero intensities leaving zero-valued pixels unchanged.
 *
 * \ingroup IntensityImageFilters  Multithreaded
 */
namespace Function {  
  
template< class TInput, class TOutput>
class LogNonZeroIntensities
{
public:
  LogNonZeroIntensities() {}
  ~LogNonZeroIntensities() {}
  bool operator!=( const LogNonZeroIntensities & ) const
    {
    return false;
    }
  bool operator==( const LogNonZeroIntensities & other ) const
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
class ITK_EXPORT LogNonZeroIntensitiesImageFilter :
    public
UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                        Function::LogNonZeroIntensities< typename TInputImage::PixelType, 
                                       typename TOutputImage::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef LogNonZeroIntensitiesImageFilter                               Self;
  typedef UnaryFunctorImageFilter<
    TInputImage,TOutputImage, 
    Function::LogNonZeroIntensities< typename TInputImage::PixelType, 
                   typename TOutputImage::PixelType> > Superclass;
  typedef SmartPointer<Self>                           Pointer;
  typedef SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(LogNonZeroIntensitiesImageFilter, 
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
  LogNonZeroIntensitiesImageFilter() {}
  virtual ~LogNonZeroIntensitiesImageFilter() {}

private:
  LogNonZeroIntensitiesImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk


#endif
