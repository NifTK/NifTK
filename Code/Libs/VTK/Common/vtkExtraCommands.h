/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef vtkExtraCommands_h
#define vtkExtraCommands_h

#include <NifTKConfigure.h>
#include <niftkVTKWin32ExportHeader.h>

#include <vtkCommand.h>

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
