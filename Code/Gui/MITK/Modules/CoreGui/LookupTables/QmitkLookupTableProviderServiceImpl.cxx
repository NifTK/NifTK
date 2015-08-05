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
std::string QmitkLookupTableProviderServiceImpl::GetName(unsigned int lookupTableIndex)
{
  return this->GetManager()->GetName(lookupTableIndex).toStdString();
}


//-----------------------------------------------------------------------------
bool QmitkLookupTableProviderServiceImpl::GetIsScaled(unsigned int lookupTableIndex)
{
  const QmitkLookupTableContainer* lutContainer = this->GetManager()->GetLookupTableContainer(lookupTableIndex);
  if (lutContainer == NULL)
  {
    mitkThrow() << "Lookup table index " << lookupTableIndex << " is invalid." << std::endl;
  }

  return lutContainer->GetIsScaled();
}


//-----------------------------------------------------------------------------
mitk::LabeledLookupTableProperty::LabelsListType
QmitkLookupTableProviderServiceImpl::GetLabels(unsigned int lookupTableIndex)
{
  const QmitkLookupTableContainer* lutContainer = this->GetManager()->GetLookupTableContainer(lookupTableIndex);
  if (lutContainer == NULL)
  {
    mitkThrow() << "Lookup table index " << lookupTableIndex << " is invalid." << std::endl;
  }

  return lutContainer->GetLabels();
}


//-----------------------------------------------------------------------------
vtkLookupTable* QmitkLookupTableProviderServiceImpl::CreateLookupTable(unsigned int lookupTableIndex,
                                                                       float lowestValueOpacity,
                                                                       float highestValueOpacity)
{
  const QmitkLookupTableContainer* lutContainer = this->GetManager()->GetLookupTableContainer(lookupTableIndex);
  if (lutContainer == NULL)
  {
    mitkThrow() << "Lookup table index " << lookupTableIndex << " is invalid." << std::endl;
  }

  vtkLookupTable *vtkLUT = vtkLookupTable::New();
  vtkLUT->DeepCopy(dynamic_cast<vtkScalarsToColors*>(const_cast<vtkLookupTable*>(lutContainer->GetLookupTable())));

  if (vtkLUT->GetNumberOfColors() == 0)
  {
    mitkThrow() << "Lookup table index " << lookupTableIndex << " has no colours." << std::endl;
  }

  if( lutContainer->GetIsScaled() )
  {
    double rgba[4];
    vtkLUT->GetTableValue(0, rgba);
    rgba[3] = lowestValueOpacity;
    vtkLUT->SetTableValue(0, rgba);

    vtkLUT->GetTableValue(vtkLUT->GetNumberOfColors()-1, rgba);
    rgba[3] = highestValueOpacity;
    vtkLUT->SetTableValue(vtkLUT->GetNumberOfColors()-1, rgba);
  }
  return vtkLUT;
}


//-----------------------------------------------------------------------------
mitk::NamedLookupTableProperty::Pointer QmitkLookupTableProviderServiceImpl::CreateLookupTableProperty(
    unsigned int lookupTableIndex,
    float lowestValueOpacity,
    float highestValueOpacity)
{
  if (lookupTableIndex >= this->GetNumberOfLookupTables())
  {
    mitkThrow() << "Lookup table index " << lookupTableIndex << " is out of range." << std::endl;
  }

  vtkLookupTable *vtkLUT = this->CreateLookupTable(lookupTableIndex, lowestValueOpacity, highestValueOpacity);

  mitk::LookupTable::Pointer mitkLUT = mitk::LookupTable::New();
  mitkLUT->SetVtkLookupTable(vtkLUT);

  mitk::NamedLookupTableProperty::Pointer mitkLUTProperty = mitk::NamedLookupTableProperty::New();
  mitkLUTProperty->SetLookupTable(mitkLUT);
  mitkLUTProperty->SetName(this->GetName(lookupTableIndex));
  mitkLUTProperty->SetIsScaled(this->GetIsScaled(lookupTableIndex));

  return mitkLUTProperty;
}


mitk::LabeledLookupTableProperty::Pointer QmitkLookupTableProviderServiceImpl::CreateLookupTableProperty(
  unsigned int lookupTableIndex )
{

  if (lookupTableIndex >= this->GetNumberOfLookupTables())
  {
    mitkThrow() << "Lookup table index " << lookupTableIndex << " is out of range." << std::endl;
  }
    vtkLookupTable *vtkLUT = this->CreateLookupTable(lookupTableIndex, 0, 0);

  mitk::LookupTable::Pointer mitkLUT = mitk::LookupTable::New();
  mitkLUT->SetVtkLookupTable(vtkLUT);

  mitk::LabeledLookupTableProperty::Pointer mitkLUTProperty = mitk::LabeledLookupTableProperty::New();
  mitkLUTProperty->SetLookupTable(mitkLUT);
  mitkLUTProperty->SetName(this->GetName(lookupTableIndex));
  mitkLUTProperty->SetIsScaled(this->GetIsScaled(lookupTableIndex));
  mitkLUTProperty->SetLabels(this->GetLabels(lookupTableIndex));

  return mitkLUTProperty;
}

bool QmitkLookupTableProviderServiceImpl::AddNewLookupTableContainer(
  QmitkLookupTableContainer* container) 
{
 
  QmitkLookupTableManager* manager = this->GetManager();
  if(manager == NULL)
    return false;

  manager->AddLookupTableContainer(container);
  return true;
}