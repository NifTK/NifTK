/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVtkInteractorStyle.h"

#include <vtkObjectFactory.h>
#include <vtkCommand.h>
#include <vtkRenderWindowInteractor.h>


namespace niftk
{

vtkStandardNewMacro(VtkInteractorStyle)

//----------------------------------------------------------------------------
VtkInteractorStyle::VtkInteractorStyle()
: mitkVtkInteractorStyle()
{
}


//----------------------------------------------------------------------------
VtkInteractorStyle::~VtkInteractorStyle()
{
}


//----------------------------------------------------------------------------
void VtkInteractorStyle::OnChar()
{
  vtkRenderWindowInteractor *rwi = this->Interactor;

  switch (rwi->GetKeyCode())
  {
  case '3' :
    break;

  default:
    Superclass::OnChar();
  }
}

}
