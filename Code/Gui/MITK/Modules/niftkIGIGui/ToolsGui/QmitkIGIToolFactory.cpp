/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkIGIToolFactory.h"
#include "QmitkIGITrackerTool.h"
#include "QmitkIGIToolGui.h"

//-----------------------------------------------------------------------------
QmitkIGIToolFactory::QmitkIGIToolFactory()
{

}


//-----------------------------------------------------------------------------
QmitkIGIToolFactory::~QmitkIGIToolFactory()
{

}

QmitkIGITool::Pointer QmitkIGIToolFactory::CreateTool(ClientDescriptorXMLBuilder& descriptor)
{
  QmitkIGITool::Pointer tool = NULL;
  const QString deviceType = descriptor.getDeviceType();

  if (deviceType == QString("Tracker"))
  {
    tool = QmitkIGITrackerTool::New();
  }
  else
  {
    // ToDo: Ultrasonix etc.
  }

  return tool;
}

//-----------------------------------------------------------------------------
QmitkIGIToolGui::Pointer QmitkIGIToolFactory::CreateGUI(QmitkIGITool* tool, const QString& prefix, const QString& postfix)
{
  QmitkIGIToolGui* toolGui = NULL;

  std::string classname = tool->GetNameOfClass();
  std::string guiClassname = prefix.toStdString() + classname + postfix.toStdString();

  std::list<itk::LightObject::Pointer> allGUIs = itk::ObjectFactoryBase::CreateAllInstance(guiClassname.c_str());
  for( std::list<itk::LightObject::Pointer>::iterator iter = allGUIs.begin();
       iter != allGUIs.end();
       ++iter )
  {
    if (toolGui == NULL)
    {
      toolGui = dynamic_cast<QmitkIGIToolGui*>( iter->GetPointer() );
    }
    else
    {
      MITK_ERROR << "There is more than one GUI for " << classname << " (several factories claim ability to produce a " << guiClassname << " ) " << std::endl;
      return NULL; // people should see and fix this error
    }
  }
  return toolGui;
}


