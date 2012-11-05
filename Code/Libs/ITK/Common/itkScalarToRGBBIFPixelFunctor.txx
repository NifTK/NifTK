/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkScalarToRGBBIFPixelFunctor_txx
#define __itkScalarToRGBBIFPixelFunctor_txx

#include "itkScalarToRGBBIFPixelFunctor.h"

#include "itkUCLMacro.h"
#include <iostream>
using namespace std;

namespace itk {

namespace Functor {  

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class TScalar>
ScalarToRGBBIFPixelFunctor<TScalar>
::ScalarToRGBBIFPixelFunctor()
{

}

  
/* -----------------------------------------------------------------------
   operator()
   ----------------------------------------------------------------------- */

template <class TScalar>
typename ScalarToRGBBIFPixelFunctor<TScalar>::RGBPixelType
ScalarToRGBBIFPixelFunctor<TScalar>
::operator()( const TScalar & v) const
{
  RGBPixelType ans;

  switch ((int) v) 
    {

      // Flat regions are pink
    case 0: {
      
      ans[0] = static_cast< RGBComponentType >( 255 );
      ans[1] = static_cast< RGBComponentType >( 153 );
      ans[2] = static_cast< RGBComponentType >( 204 );
      
      break;
    }

      // Slope-like regions are grey
    case 1: {
      
      ans[0] = static_cast< RGBComponentType >( 204 );
      ans[1] = static_cast< RGBComponentType >( 204 );
      ans[2] = static_cast< RGBComponentType >( 204 );
      
      break;
    }

      // Maxima are black
    case 2: {
      
      ans[0] = static_cast< RGBComponentType >( 0 );
      ans[1] = static_cast< RGBComponentType >( 0 );
      ans[2] = static_cast< RGBComponentType >( 0 );
      
      break;
    }

      // Minima are white
    case 3: {
      
      ans[0] = static_cast< RGBComponentType >( 255 );
      ans[1] = static_cast< RGBComponentType >( 255 );
      ans[2] = static_cast< RGBComponentType >( 255 );
      
      break;
    }

      // Dark lines are blue
    case 4: {
      
      ans[0] = static_cast< RGBComponentType >(  26 );
      ans[1] = static_cast< RGBComponentType >(  26 );
      ans[2] = static_cast< RGBComponentType >( 192 );
      
      break;
    }

      // Light lines are yellow
    case 5: {
      
      ans[0] = static_cast< RGBComponentType >( 244 );
      ans[1] = static_cast< RGBComponentType >( 244 );
      ans[2] = static_cast< RGBComponentType >(   0 );
      
      break;
    }

      // Saddles are green
    case 6: {
      
      ans[0] = static_cast< RGBComponentType >(   0 );
      ans[1] = static_cast< RGBComponentType >( 154 );
      ans[2] = static_cast< RGBComponentType >(   0 );
      
      break;
    }
      
      // Unrecognised codes are red
    default: {
      
      ans[0] = static_cast< RGBComponentType >( 168 );
      ans[1] = static_cast< RGBComponentType >(   0 );
      ans[2] = static_cast< RGBComponentType >(   3 );
      
      std::cerr << "BIF code must satisfy: 0 < n <= 6";
    }
  }

  return ans;
}


  


} // end namespace Functor

} // end namespace itk


#endif
