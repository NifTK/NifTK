/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASDataStorageEditorInput_h
#define mitkMIDASDataStorageEditorInput_h

#include <berryIEditorInput.h>
#include <mitkIDataStorageReference.h>
#include <uk_ac_ucl_cmic_midaseditor_Export.h>

namespace mitk
{

/**
 * \class MIDASDataStorageEditorInput
 * \brief Used to connect the editor to the data-storage, and provide a key such that
 * when a data image in the data storage is opened, the application framework knows that
 * this is a valid type of editor with which to view the data.
 * \ingroup uk_ac_ucl_cmic_midaseditor
 */
class MIDASEDITOR_EXPORT MIDASDataStorageEditorInput : public berry::IEditorInput
{
public:
  berryObjectMacro(MIDASDataStorageEditorInput);

  MIDASDataStorageEditorInput();
  MIDASDataStorageEditorInput(IDataStorageReference::Pointer ref);

  bool Exists() const;
  std::string GetName() const;
  std::string GetToolTipText() const;

  IDataStorageReference::Pointer GetDataStorageReference();

  bool operator==(const berry::Object*) const;

private:

  IDataStorageReference::Pointer m_DataStorageRef;
};

}

#endif
