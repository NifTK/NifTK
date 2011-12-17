/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-02 17:25:37 +0100 (Thu, 02 Sep 2010) $
 Revision          : $Revision: 6276 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef WIDGETHELPER_CPP
#define WIDGETHELPER_CPP

int ConvertSpinBoxValueToSliderValue(double spinBoxValue, double spinBoxMin, double spinBoxMax, int sliderMin, int sliderMax)
{
  double fraction = (spinBoxValue-spinBoxMin)/(spinBoxMax-spinBoxMin);
  int result = (int)(fraction*((double)(sliderMax - sliderMin)) + sliderMin);
  return result;
}

double ConvertSliderValueToSpinBoxValue(int sliderValue, double spinBoxMin, double spinBoxMax, int sliderMin, int sliderMax)
{
  double fraction = ((double)(sliderValue - sliderMin))/((double)(sliderMax-sliderMin));
  double result = fraction*(spinBoxMax-spinBoxMin) + spinBoxMin;
  return result;
}

#endif
