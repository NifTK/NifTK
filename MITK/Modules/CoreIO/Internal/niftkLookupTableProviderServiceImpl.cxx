/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkLookupTableProviderServiceImpl_p.h"

#include <iostream>

#include <vtkLookupTable.h>

#include <mitkExceptionMacro.h>
#include <mitkLookupTable.h>

#include "niftkLookupTableContainer.h"
#include "niftkLookupTableManager.h"


namespace niftk
{

//-----------------------------------------------------------------------------
LookupTableProviderServiceImpl::LookupTableProviderServiceImpl()
{
  m_Manager.reset(NULL);
}


//-----------------------------------------------------------------------------
LookupTableManager* LookupTableProviderServiceImpl::GetManager()
{
  if (m_Manager.get() == NULL)
  {
    m_Manager.reset(new LookupTableManager);
  }
  return m_Manager.get();
}


//-----------------------------------------------------------------------------
LookupTableProviderServiceImpl::~LookupTableProviderServiceImpl()
{
}


//-----------------------------------------------------------------------------
unsigned int LookupTableProviderServiceImpl::GetNumberOfLookupTables()
{
  return this->GetManager()->GetNumberOfLookupTables();
}


//-----------------------------------------------------------------------------
bool LookupTableProviderServiceImpl::CheckName(const QString& name)
{
  return this->GetManager()->CheckName(name);
}


//-----------------------------------------------------------------------------
std::vector<QString>LookupTableProviderServiceImpl::GetTableNames()
{
  return this->GetManager()->GetTableNames();
}


//-----------------------------------------------------------------------------
bool LookupTableProviderServiceImpl::GetIsScaled(const QString& lookupTableName)
{
  const LookupTableContainer* lutContainer = this->GetManager()->GetLookupTableContainer(lookupTableName);
  if (lutContainer == NULL)
  {
    mitkThrow() << "Lookup table name " << lookupTableName.toStdString().c_str() << " is invalid." << std::endl;
  }

  return lutContainer->GetIsScaled();
}


//-----------------------------------------------------------------------------
niftk::LabeledLookupTableProperty::LabelListType
LookupTableProviderServiceImpl::GetLabels(const QString& lookupTableName)
{
  const LookupTableContainer* lutContainer = this->GetManager()->GetLookupTableContainer(lookupTableName);
  if (lutContainer == NULL)
  {
    mitkThrow() << "Lookup table name " << lookupTableName.toStdString().c_str() << " is invalid." << std::endl;
  }

  return lutContainer->GetLabels();
}


//-----------------------------------------------------------------------------
vtkLookupTable* LookupTableProviderServiceImpl
::CreateLookupTable(const QString& lookupTableName, float lowestValueOpacity, float highestValueOpacity)
{
  const LookupTableContainer* lutContainer = this->GetManager()->GetLookupTableContainer(lookupTableName);
  if (lutContainer == NULL)
  {
    mitkThrow() << "Lookup table name " << lookupTableName.toStdString().c_str() << " is invalid." << std::endl;
  }

  vtkLookupTable *vtkLUT = vtkLookupTable::New();
  vtkLUT->DeepCopy(dynamic_cast<vtkScalarsToColors*>(const_cast<vtkLookupTable*>(lutContainer->GetLookupTable())));

  if (vtkLUT->GetNumberOfColors() == 0)
  {
    mitkThrow() << "Lookup table " << lookupTableName.toStdString().c_str() << " has no colours." << std::endl;
  }

  if (lutContainer->GetIsScaled())
  {
    double rgba[4];
    vtkLUT->GetTableValue(0, rgba);
    rgba[3] = lowestValueOpacity;
    vtkLUT->SetTableValue(0, rgba);

    vtkLUT->GetTableValue(vtkLUT->GetNumberOfColors() - 1, rgba);
    rgba[3] = highestValueOpacity;
    vtkLUT->SetTableValue(vtkLUT->GetNumberOfColors() - 1, rgba);
  }
  return vtkLUT;
}


//-----------------------------------------------------------------------------
niftk::NamedLookupTableProperty::Pointer
LookupTableProviderServiceImpl
::CreateLookupTableProperty(const QString& lookupTableName, float lowestValueOpacity, float highestValueOpacity)
{
  vtkLookupTable *vtkLUT = this->CreateLookupTable(lookupTableName, lowestValueOpacity, highestValueOpacity);

  mitk::LookupTable::Pointer mitkLUT = mitk::LookupTable::New();
  mitkLUT->SetVtkLookupTable(vtkLUT);

  niftk::NamedLookupTableProperty::Pointer mitkLUTProperty = niftk::NamedLookupTableProperty::New();
  mitkLUTProperty->SetLookupTable(mitkLUT);
  mitkLUTProperty->SetName(lookupTableName.toStdString());
  mitkLUTProperty->SetIsScaled(this->GetIsScaled(lookupTableName));

  return mitkLUTProperty;
}


//-----------------------------------------------------------------------------
niftk::LabeledLookupTableProperty::Pointer
LookupTableProviderServiceImpl::CreateLookupTableProperty(const QString& lookupTableName)
{
  vtkLookupTable *vtkLUT = this->CreateLookupTable(lookupTableName, 0, 0);

  mitk::LookupTable::Pointer mitkLUT = mitk::LookupTable::New();
  mitkLUT->SetVtkLookupTable(vtkLUT);

  niftk::LabeledLookupTableProperty::Pointer mitkLUTProperty = niftk::LabeledLookupTableProperty::New();
  mitkLUTProperty->SetLookupTable(mitkLUT);
  mitkLUTProperty->SetName(lookupTableName.toStdString());
  mitkLUTProperty->SetIsScaled(this->GetIsScaled(lookupTableName));
  mitkLUTProperty->SetLabels(this->GetLabels(lookupTableName));

  return mitkLUTProperty;
}


//-----------------------------------------------------------------------------
void LookupTableProviderServiceImpl::AddNewLookupTableContainer(const LookupTableContainer* container)
{
  LookupTableManager* manager = this->GetManager();
  if (manager == NULL)
  {
    return;
  }

  manager->AddLookupTableContainer(container);
}


//-----------------------------------------------------------------------------
void LookupTableProviderServiceImpl
::ReplaceLookupTableContainer(const LookupTableContainer* container, const QString& lookupTableName)
{
  LookupTableManager* manager = this->GetManager();
  if (manager == NULL)
  {
    return;
  }

  manager->ReplaceLookupTableContainer(container, container->GetDisplayName());
}

}
