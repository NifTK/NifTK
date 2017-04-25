/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

/**
 * \brief Overrides mitkVtkInteractorStyle to supress some basic interactions implemented in VTK.
 */

#ifndef niftkVtkInteractorStyle_h
#define niftkVtkInteractorStyle_h

#include <mitkVtkInteractorStyle.h>

namespace niftk
{

class VtkInteractorStyle : public mitkVtkInteractorStyle
{
public:

  // default VTK c'tor
  static VtkInteractorStyle *New();
  vtkTypeMacro(VtkInteractorStyle, mitkVtkInteractorStyle)

  // Description:
  // OnChar is triggered when an ASCII key is pressed.
  // Overrides superclass implementation to supress stereo mode.
  virtual void OnChar() override;

protected:

  VtkInteractorStyle();
  ~VtkInteractorStyle();


private:
  VtkInteractorStyle(const VtkInteractorStyle&);  // Not implemented.
  void operator=(const VtkInteractorStyle&);  // Not implemented.
};

}

#endif
