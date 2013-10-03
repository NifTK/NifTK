/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASDataStorageEditorInput.h"

#include <berryPlatform.h>
#include <mitkIDataStorageService.h>

namespace mitk
{

//-----------------------------------------------------------------------------
MIDASDataStorageEditorInput::MIDASDataStorageEditorInput()
{
}


//-----------------------------------------------------------------------------
MIDASDataStorageEditorInput::MIDASDataStorageEditorInput(IDataStorageReference::Pointer ref)
{
  m_DataStorageRef = ref;
}


//-----------------------------------------------------------------------------
bool MIDASDataStorageEditorInput::Exists() const
{
  return true;
}


//-----------------------------------------------------------------------------
std::string MIDASDataStorageEditorInput::GetName() const
{
  return "MIDAS DataStorage Scene";
}


//-----------------------------------------------------------------------------
std::string MIDASDataStorageEditorInput::GetToolTipText() const
{
  return "";
}


//-----------------------------------------------------------------------------
bool MIDASDataStorageEditorInput::operator==(const berry::Object* o) const
{
  if (const MIDASDataStorageEditorInput* input = dynamic_cast<const MIDASDataStorageEditorInput*>(o))
    return this->GetName() == input->GetName();

  return false;
}


//-----------------------------------------------------------------------------
IDataStorageReference::Pointer MIDASDataStorageEditorInput::GetDataStorageReference()
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

} // end namespace
