/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-01 19:03:07 +0100 (Fri, 01 Jul 2011) $
 Revision          : $Revision: 6628 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef VTKEXTRACOMMANDS_H
#define VTKEXTRACOMMANDS_H

#include "NifTKConfigure.h"
#include "niftkVTKWin32ExportHeader.h"

#include "vtkCommand.h"

/**
 * \class vtkExtraCommands
 * \brief Pretty dumb class (TODO, maybe an alternative), to hold additional enums for creating command callbacks.
 */
class NIFTKVTK_WINEXPORT vtkExtraCommands : public vtkCommand
{

public:

  enum ExtraEventIds
  {
    BeforeCameraMoves = vtkCommand::UserEvent,
    AfterCameraMoves,
    LeftButtonUp,
    MouseMoved
  };

protected:

  vtkExtraCommands();
  virtual ~vtkExtraCommands() {}
  vtkExtraCommands(const vtkExtraCommands& c) : vtkCommand(c) {}
  void operator=(const vtkExtraCommands&) {}

private:

};
#endif
