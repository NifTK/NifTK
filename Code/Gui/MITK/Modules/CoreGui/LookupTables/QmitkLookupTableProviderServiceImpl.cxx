/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkLookupTableProviderServiceImpl_p.h"
#include "QmitkLookupTableManager.h"
#include "QmitkLookupTableContainer.h"
#include <iostream>
#include <vtkLookupTable.h>
#include <mitkExceptionMacro.h>
#include <mitkLookupTable.h>


//-----------------------------------------------------------------------------
QmitkLookupTableProviderServiceImpl::QmitkLookupTableProviderServiceImpl()
{
  m_Manager.reset(NULL);
}


//-----------------------------------------------------------------------------
QmitkLookupTableManager* QmitkLookupTableProviderServiceImpl::GetManager()
{
  if (m_Manager.get() == NULL)
  {
    m_Manager.reset(new QmitkLookupTableManager);
  }
  return m_Manager.get();
}


//-----------------------------------------------------------------------------
QmitkLookupTableProviderServiceImpl::~QmitkLookupTableProviderServiceImpl()
{
}


//-----------------------------------------------------------------------------
unsigned int QmitkLookupTableProviderServiceImpl::GetNumberOfLookupTables()
{
  return this->GetManager()->GetNumberOfLookupTables();
}


//-----------------------------------------------------------------------------
bool QmitkLookupTableProviderServiceImpl::CheckName(const QString& name)
{
  return this->GetManager()->CheckName(name);
}


//-----------------------------------------------------------------------------
std::vector<QString>QmitkLookupTableProviderServiceImpl::GetTableNames()
{
  return this->GetManager()->GetTableNames();
}


//-----------------------------------------------------------------------------
bool QmitkLookupTableProviderServiceImpl::GetIsScaled(const QString& lookupTableName)
{
  const QmitkLookupTableContainer* lutContainer = this->GetManager()->GetLookupTableContainer(lookupTableName);
  if (lutContainer == NULL)
  {
    mitkThrow() << "Lookup table name " << lookupTableName.toStdString().c_str() << " is invalid." << std::endl;
  }

  return lutContainer->GetIsScaled();
}


//-----------------------------------------------------------------------------
mitk::LabeledLookupTableProperty::LabelListType
QmitkLookupTableProviderServiceImpl::GetLabels(const QString& lookupTableName)
{
  const QmitkLookupTableContainer* lutContainer = this->GetManager()->GetLookupTableContainer(lookupTableName);
  if (lutContainer == NULL)
  {
    mitkThrow() << "Lookup table name " << lookupTableName.toStdString().c_str() << " is invalid." << std::endl;
  }

  return lutContainer->GetLabels();
}


//-----------------------------------------------------------------------------
vtkLookupTable* QmitkLookupTableProviderServiceImpl
::CreateLookupTable(const QString& lookupTableName, float lowestValueOpacity, float highestValueOpacity)
{
  const QmitkLookupTableContainer* lutContainer = this->GetManager()->GetLookupTableContainer(lookupTableName);
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
mitk::NamedLookupTableProperty::Pointer 
QmitkLookupTableProviderServiceImpl
::CreateLookupTableProperty(const QString& lookupTableName, float lowestValueOpacity, float highestValueOpacity)
{
  vtkLookupTable *vtkLUT = this->CreateLookupTable(lookupTableName, lowestValueOpacity, highestValueOpacity);

  mitk::LookupTable::Pointer mitkLUT = mitk::LookupTable::New();
  mitkLUT->SetVtkLookupTable(vtkLUT);

  mitk::NamedLookupTableProperty::Pointer mitkLUTProperty = mitk::NamedLookupTableProperty::New();
  mitkLUTProperty->SetLookupTable(mitkLUT);
  mitkLUTProperty->SetName(lookupTableName.toStdString());
  mitkLUTProperty->SetIsScaled(this->GetIsScaled(lookupTableName));

  return mitkLUTProperty;
}


//-----------------------------------------------------------------------------
mitk::LabeledLookupTableProperty::Pointer 
QmitkLookupTableProviderServiceImpl::CreateLookupTableProperty(const QString& lookupTableName)
{
  vtkLookupTable *vtkLUT = this->CreateLookupTable(lookupTableName, 0, 0);

  mitk::LookupTable::Pointer mitkLUT = mitk::LookupTable::New();
  mitkLUT->SetVtkLookupTable(vtkLUT);

  mitk::LabeledLookupTableProperty::Pointer mitkLUTProperty = mitk::LabeledLookupTableProperty::New();
  mitkLUTProperty->SetLookupTable(mitkLUT);
  mitkLUTProperty->SetName(lookupTableName.toStdString());
  mitkLUTProperty->SetIsScaled(this->GetIsScaled(lookupTableName));
  mitkLUTProperty->SetLabels(this->GetLabels(lookupTableName));

  return mitkLUTProperty;
}


//-----------------------------------------------------------------------------
void QmitkLookupTableProviderServiceImpl::AddNewLookupTableContainer(const QmitkLookupTableContainer* container) 
{
  QmitkLookupTableManager* manager = this->GetManager();
  if (manager == NULL)
  {
    return;
  }
  
  manager->AddLookupTableContainer(container);
}


//-----------------------------------------------------------------------------
void QmitkLookupTableProviderServiceImpl
::ReplaceLookupTableContainer(const QmitkLookupTableContainer* container, const QString& lookupTableName) 
{
  QmitkLookupTableManager* manager = this->GetManager();
  if (manager == NULL)
  {
    return;
  }
  
  manager->ReplaceLookupTableContainer(container, container->GetDisplayName());
}