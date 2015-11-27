/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkLookupTableContainer.h"

//-----------------------------------------------------------------------------
QmitkLookupTableContainer::QmitkLookupTableContainer(const vtkLookupTable* lut)
: m_Order(-1)
, m_IsScaled(true)
, m_DisplayName("None")
{
  vtkLookupTable *newTable = vtkLookupTable::New();
  newTable->DeepCopy(const_cast<vtkLookupTable*>(lut));
  newTable->SetNanColor(const_cast<vtkLookupTable*>(lut)->GetNanColor());

  m_LookupTable = const_cast<vtkLookupTable const*>(newTable);
}


//-----------------------------------------------------------------------------
QmitkLookupTableContainer::QmitkLookupTableContainer(const vtkLookupTable* lut, const LabelListType& labels)
{
  vtkLookupTable *newTable = vtkLookupTable::New();
  newTable->DeepCopy(const_cast<vtkLookupTable*>(lut));
  newTable->SetNanColor(const_cast<vtkLookupTable*>(lut)->GetNanColor());

  m_LookupTable = const_cast<vtkLookupTable const*>(newTable);

  m_DisplayName = QString("");
  m_Order = 0;
  m_IsScaled = false;
  m_Labels = labels;
}


//-----------------------------------------------------------------------------
QmitkLookupTableContainer::~QmitkLookupTableContainer()
{
  if (m_LookupTable != NULL)
  {
    vtkLookupTable *nonConst = const_cast<vtkLookupTable*>(m_LookupTable);
    nonConst->Delete();
    nonConst = NULL;
  }
}
