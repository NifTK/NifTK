/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkScalarToRGBOBIFPixelFunctor.h,v $
  Language:  C++
  Date:      $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
  Version:   $Revision: 7333 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkScalarToRGBOBIFPixelFunctor_h
#define __itkScalarToRGBOBIFPixelFunctor_h

#include "itkRGBPixel.h"

namespace itk {

namespace Functor {  

/**
 * \class ScalarToRGBOBIFPixelFunctor
 * \brief Function object which maps a scalar value into an RGB pixel
 * value for basic image feature images according to Lewis Griffin's
 * colour scheme.
 */

template< class TScalar, int TNumberOfOrientations >
class ITK_EXPORT ScalarToRGBOBIFPixelFunctor
{

public:

  ScalarToRGBOBIFPixelFunctor();
  ~ScalarToRGBOBIFPixelFunctor() {};

  typedef unsigned char               RGBComponentType;
  typedef RGBPixel<RGBComponentType>  RGBPixelType;
  typedef TScalar                     ScalarType;
  
  RGBPixelType operator()( const TScalar &) const;
 
  void InterpolateRGB( const int c1[3], const int c2[3], 
                       double fraction,
                       RGBPixelType &ans ) const;
 
private:

  static const int OBIFcolors[23][3];

};
  
} // end namespace functor

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkScalarToRGBOBIFPixelFunctor.txx"
#endif

#endif
