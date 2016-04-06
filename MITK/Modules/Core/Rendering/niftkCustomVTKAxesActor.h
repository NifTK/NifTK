/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#ifndef niftkCustomVTKAxesActor_h
#define niftkCustomVTKAxesActor_h

#include "niftkCoreExports.h"

#include <vtkAxesActor.h>
#include <vtkCaptionActor2D.h>

namespace niftk
{

class NIFTKCORE_EXPORT CustomVTKAxesActor : public vtkAxesActor
{
  public:
   inline void SetAxisLabelWidth(double w) {this->XAxisLabel->SetWidth(w); this->YAxisLabel->SetWidth(w); this->ZAxisLabel->SetWidth(w);}
   inline double GetAxisLabelWidth() {return m_AxesLabelWidth;}
   inline void SetAxisLabelHeight(double h) {this->XAxisLabel->SetHeight(h); this->YAxisLabel->SetHeight(h); this->ZAxisLabel->SetHeight(h);}
   inline double GetAxisLabelHeight() {return m_AxesLabelHeight;}

   CustomVTKAxesActor();

 private:
   double m_AxesLabelWidth;
   double m_AxesLabelHeight;
};

}
 
#endif
 