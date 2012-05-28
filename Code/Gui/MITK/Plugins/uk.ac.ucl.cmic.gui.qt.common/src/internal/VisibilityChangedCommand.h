/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 15:53:45 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7857 $
 Last modified by  : $Author: mjc $

 Original author   : Miklos Espak <espakm@gmail.com>

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

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
