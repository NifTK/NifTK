/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkScalarToRGBBIFPixelFunctor.h,v $
  Language:  C++
  Date:      $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
  Version:   $Revision: 7333 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkScalarToRGBBIFPixelFunctor_h
#define __itkScalarToRGBBIFPixelFunctor_h

#include "itkRGBPixel.h"

namespace itk {

namespace Functor {  

/**
 * \class ScalarToRGBBIFPixelFunctor
 * \brief Function object which maps a scalar value into an RGB pixel
 * value for basic image feature images according to Lewis Griffin's
 * colour scheme.
 */

template< class TScalar >
class ITK_EXPORT ScalarToRGBBIFPixelFunctor
{
public:
  ScalarToRGBBIFPixelFunctor();
  ~ScalarToRGBBIFPixelFunctor() {};

  typedef unsigned char               RGBComponentType;
  typedef RGBPixel<RGBComponentType>  RGBPixelType;
  typedef TScalar                     ScalarType;
  
  RGBPixelType operator()( const TScalar &) const;
  
private:

};
  
} // end namespace functor

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkScalarToRGBBIFPixelFunctor.txx"
#endif

#endif
