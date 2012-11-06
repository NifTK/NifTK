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

#ifndef __itkScalarToRGBOBIFPixelFunctor_txx
#define __itkScalarToRGBOBIFPixelFunctor_txx

#include "itkScalarToRGBOBIFPixelFunctor.h"

#include "itkUCLMacro.h"
#include <iostream>
using namespace std;



namespace itk {

namespace Functor {  


template <class TScalar, int TNumberOfOrientations>
const int 
ScalarToRGBOBIFPixelFunctor<TScalar, TNumberOfOrientations>
::OBIFcolors[23][3] = {
  {127, 127, 127},		// 0  Flat

  {128, 160, 255},		// 1  Slope 1       
  {255, 128, 223},		// 2  Slope 2       
  {255, 223, 128},		// 3  Slope 3       
  {128, 255, 160},		// 4  Slope 4       
  { 64,  80, 127},		// 5  Slope 5       
  {127,  64, 111},		// 6  Slope 6       
  {127, 111,  64},		// 7  Slope 7       
  { 64, 127,  80},		// 8  Slope 8       

  {  0,   0,   0},		// 9  Minimum       
  {255, 255, 255},		// 10 Maximum       

  {255, 191,   0},		// 11 Light line 1  
  {  0, 255,  64},		// 12 Light line 2  
  {  0,  64, 255},		// 13 Light line 3  
  {255,   0, 191},		// 14 Light line 4  

  {127,  95,   0},		// 15 Dark line 1   
  {  0, 127,  32},		// 16 Dark line 2   
  {  0,  32, 127},		// 17 Dark line 3   
  {127,   0,  95},		// 18 Dark line 4   

  {255,   0,   0},		// 19 Saddle 1      
  {128, 255,   0},		// 20 Saddle 2      
  {  0, 255, 255},		// 21 Saddle 3      
  {128,   0, 255}		// 22 Saddle 4      
};

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class TScalar, int TNumberOfOrientations>
ScalarToRGBOBIFPixelFunctor<TScalar, TNumberOfOrientations>
::ScalarToRGBOBIFPixelFunctor()
{

}


/* -----------------------------------------------------------------------
   InterpolateRGB()
   ----------------------------------------------------------------------- */

template <class TScalar, int TNumberOfOrientations>
void
ScalarToRGBOBIFPixelFunctor<TScalar, TNumberOfOrientations>
::InterpolateRGB( const int c1[3], const int c2[3], 
                  double fraction,
                  RGBPixelType &ans ) const
{
  ans[0] = static_cast<RGBComponentType>( (1. - fraction)*c1[0] + fraction*c2[0] );
  ans[1] = static_cast<RGBComponentType>( (1. - fraction)*c1[1] + fraction*c2[1] );
  ans[2] = static_cast<RGBComponentType>( (1. - fraction)*c1[2] + fraction*c2[2] );
}


/* -----------------------------------------------------------------------
   operator()
   ----------------------------------------------------------------------- */

template <class TScalar, int TNumberOfOrientations>
typename ScalarToRGBOBIFPixelFunctor<TScalar, TNumberOfOrientations>::RGBPixelType
ScalarToRGBOBIFPixelFunctor<TScalar, TNumberOfOrientations>
::operator()( const TScalar &v ) const
{
  RGBPixelType ans;

  if ( v == 0 ) {               // Flat

    ans[0] = OBIFcolors[0][0];
    ans[1] = OBIFcolors[0][1];
    ans[2] = OBIFcolors[0][2];
  }
   
  else if ( v <= TNumberOfOrientations ) { // Slope

    double orientation = 8.*(v - 1.)/((double) TNumberOfOrientations);
    double sector = floor( orientation );
    double fraction = orientation - sector;

    if ( sector < 7 )
      InterpolateRGB( OBIFcolors[(int) sector + 1], OBIFcolors[(int) sector + 2],
                      fraction, ans );
    else
      InterpolateRGB( OBIFcolors[(int) sector + 1], OBIFcolors[1],
                      fraction, ans );
  }
 
  else if ( v == TNumberOfOrientations + 1 ) { // Minimum

    ans[0] = OBIFcolors[9][0];
    ans[1] = OBIFcolors[9][1];
    ans[2] = OBIFcolors[9][2];
  }

  else if ( v == TNumberOfOrientations + 2 ) { // Maximum

    ans[0] = OBIFcolors[10][0];
    ans[1] = OBIFcolors[10][1];
    ans[2] = OBIFcolors[10][2];
  }
 
  else if ( (v > TNumberOfOrientations + 2) &&
            (v < TNumberOfOrientations + 3 + TNumberOfOrientations/2) ) { // Light line
    
    double start = TNumberOfOrientations + 3;
    double orientation = 4.*(v - start)/((double) TNumberOfOrientations/2.);
    double sector = floor( orientation );
    double fraction = orientation - sector;

    if ( sector < 3 )
      InterpolateRGB( OBIFcolors[(int) sector + 11], OBIFcolors[(int) sector + 12],
                      fraction, ans );
    else
      InterpolateRGB( OBIFcolors[(int) sector + 11], OBIFcolors[11],
                      fraction, ans );
  }
 
  else if ( (v > TNumberOfOrientations + 2 + TNumberOfOrientations/2) &&
            (v < TNumberOfOrientations + 3 + TNumberOfOrientations) ) { // Dark line

    double start = 3*TNumberOfOrientations/2 + 3;
    double orientation = 4.*(v - start)/((double) TNumberOfOrientations/2.);
    double sector = floor( orientation );
    double fraction = orientation - sector;

    if ( sector < 3 )
      InterpolateRGB( OBIFcolors[(int) sector + 15], OBIFcolors[(int) sector + 16],
                      fraction, ans );
    else
      InterpolateRGB( OBIFcolors[(int) sector + 15], OBIFcolors[15],
                      fraction, ans );
  }
 
  else if ( (v > TNumberOfOrientations + 2 + TNumberOfOrientations) &&
	    (v < 2*TNumberOfOrientations + 3 + TNumberOfOrientations/2) ) { // Saddle

    double start = 2*TNumberOfOrientations + 3;
    double orientation = 4.*(v - start)/((double) TNumberOfOrientations/2.);
    double sector = floor( orientation );
    double fraction = orientation - sector;

    if ( sector < 3 )
      InterpolateRGB( OBIFcolors[(int) sector + 19], OBIFcolors[(int) sector + 20],
                      fraction, ans );
    else
      InterpolateRGB( OBIFcolors[(int) sector + 19], OBIFcolors[19],
                      fraction, ans );
  }
  
  else {                        // Unrecognised codes are black
      
    ans[0] = 0;
    ans[1] = 0;
    ans[2] = 0;
      
    std::cerr << "BIF code must satisfy: 0 < n <= 22";
  }

  return ans;
}


  


} // end namespace Functor

} // end namespace itk


#endif
