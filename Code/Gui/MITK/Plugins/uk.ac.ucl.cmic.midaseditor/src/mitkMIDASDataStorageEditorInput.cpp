/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-08 06:49:39 +0100 (Sat, 08 Oct 2011) $
 Revision          : $Revision: 7466 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASDataStorageEditorInput.h"

#include <berryPlatform.h>
#include <mitkIDataStorageService.h>

namespace mitk
{

MIDASDataStorageEditorInput::MIDASDataStorageEditorInput()
{
}

MIDASDataStorageEditorInput::MIDASDataStorageEditorInput(IDataStorageReference::Pointer ref)
{
  m_DataStorageRef = ref;
}

bool MIDASDataStorageEditorInput::Exists() const
{
  return true;
}

std::string MIDASDataStorageEditorInput::GetName() const
{
  return "MIDAS DataStorage Scene";
}

std::string MIDASDataStorageEditorInput::GetToolTipText() const
{
  return "";
}

bool MIDASDataStorageEditorInput::operator==(const berry::Object* o) const
{
  if (const MIDASDataStorageEditorInput* input = dynamic_cast<const MIDASDataStorageEditorInput*>(o))
    return this->GetName() == input->GetName();

  return false;
}

IDataStorageReference::Pointer
MIDASDataStorageEditorInput::GetDataStorageReference()
{
  if (m_DataStorageRef.IsNull())
  {
    berry::ServiceRegistry& serviceRegistry = berry::Platform::GetServiceRegistry();
    IDataStorageService::Pointer dataService = serviceRegistry.GetServiceById<IDataStorageService>(IDataStorageService::ID);
    if (!dataService) return IDataStorageReference::Pointer(0);
    m_DataStorageRef = dataService->GetDefaultDataStorage();
  }

  return m_DataStorageRef;
}

}
