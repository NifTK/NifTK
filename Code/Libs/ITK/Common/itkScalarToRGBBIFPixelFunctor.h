/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
