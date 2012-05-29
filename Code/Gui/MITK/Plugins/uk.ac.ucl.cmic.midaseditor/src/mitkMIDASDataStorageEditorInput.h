/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKMIDASDATASTORAGEEDITORINPUT_H_
#define MITKMIDASDATASTORAGEEDITORINPUT_H_

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

#endif /*MITKMIDASDATASTORAGEEDITORINPUT_H_*/
