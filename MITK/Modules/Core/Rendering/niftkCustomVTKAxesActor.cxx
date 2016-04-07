/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCustomVTKAxesActor.h"
#include <vtkTextProperty.h>

niftk::CustomVTKAxesActor::CustomVTKAxesActor() 
: vtkAxesActor() 
{
  //default: 0.25
  m_AxesLabelWidth = 0.1; 
  this->XAxisLabel->SetWidth(0.1);
  this->YAxisLabel->SetWidth(0.1);
  this->ZAxisLabel->SetWidth(0.1);

  //default: 0.1
  m_AxesLabelHeight = 0.04;
  this->XAxisLabel->SetHeight(0.05);
  this->YAxisLabel->SetHeight(0.05);
  this->ZAxisLabel->SetHeight(0.05);

  vtkTextProperty* tprop = this->XAxisLabel->GetCaptionTextProperty();
  tprop->ItalicOff();
  tprop->BoldOff();
  tprop->ShadowOn();
  this->XAxisLabel->SetCaptionTextProperty(tprop);

  tprop = this->YAxisLabel->GetCaptionTextProperty();
  tprop->ItalicOff();
  tprop->BoldOff();
  tprop->ShadowOn();
  this->YAxisLabel->SetCaptionTextProperty(tprop);

  tprop = this->ZAxisLabel->GetCaptionTextProperty();
  tprop->ItalicOff();
  tprop->BoldOff();
  tprop->ShadowOn();
  this->ZAxisLabel->SetCaptionTextProperty(tprop);
}
