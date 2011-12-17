/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-02 17:25:37 +0100 (Thu, 02 Sep 2010) $
 Revision          : $Revision: 6628 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef WIDGETHELPER_H
#define WIDGETHELPER_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

extern "C++" NIFTKQT_WINEXPORT int ConvertSpinBoxValueToSliderValue(double spinBoxValue, double spinBoxMin, double spinBoxMax, int sliderMin, int sliderMax);

extern "C++" NIFTKQT_WINEXPORT double ConvertSliderValueToSpinBoxValue(int sliderValue, double spinBoxMin, double spinBoxMax, int sliderMin, int sliderMax);

#endif
