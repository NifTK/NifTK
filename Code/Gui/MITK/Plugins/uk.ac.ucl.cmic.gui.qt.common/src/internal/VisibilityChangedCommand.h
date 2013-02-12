/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __VisibilityChangedCommand_h
#define __VisibilityChangedCommand_h

#include <itkCommand.h>

#include <mitkCommon.h>

namespace mitk {
class DataNode;
}

class VisibilityChangeObserver;

class VisibilityChangedCommand : public itk::Command
{
public:
  mitkClassMacro(VisibilityChangedCommand, itk::Command);
  mitkNewMacro2Param(VisibilityChangedCommand, VisibilityChangeObserver*, const mitk::DataNode*);

  VisibilityChangedCommand(VisibilityChangeObserver* observer, const mitk::DataNode* node);
  virtual ~VisibilityChangedCommand();

  virtual void Execute(itk::Object* /*caller*/, const itk::EventObject& /*event*/);
  virtual void Execute(const itk::Object* /*caller*/, const itk::EventObject& /*event*/);

private:
  VisibilityChangeObserver* m_Observer;
  const mitk::DataNode* m_Node;
};

#endif
