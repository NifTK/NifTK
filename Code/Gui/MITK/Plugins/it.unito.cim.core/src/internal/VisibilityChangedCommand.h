/*=============================================================================

 KMaps:     An image processing toolkit for DCE-MRI analysis developed
            at the Molecular Imaging Center at University of Torino.

 See:       http://www.cim.unito.it

 Author:    Miklos Espak <espakm@gmail.com>

 Copyright (c) Miklos Espak
 All Rights Reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
