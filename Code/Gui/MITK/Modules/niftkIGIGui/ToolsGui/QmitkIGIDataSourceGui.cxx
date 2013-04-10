/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIDataSourceGui.h"
#include <iostream>

//-----------------------------------------------------------------------------
QmitkIGIDataSourceGui::QmitkIGIDataSourceGui()
{

}


//-----------------------------------------------------------------------------
QmitkIGIDataSourceGui::~QmitkIGIDataSourceGui()
{
  m_ReferenceCountLock.Lock();
  m_ReferenceCount = 0; // otherwise ITK will complain in LightObject's destructor
  m_ReferenceCountLock.Unlock();

  // We don't own m_Tool or m_StdMultiWidget, so don't delete them.
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceGui::Register() const
{
  // empty on purpose, just don't let ITK handle calls to Register()
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceGui::UnRegister() const
{
  // empty on purpose, just don't let ITK handle calls to UnRegister()
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceGui::SetReferenceCount(int)
{
  // empty on purpose, just don't let ITK handle calls to SetReferenceCount()
}

//-----------------------------------------------------------------------------
void QmitkIGIDataSourceGui::SetDataSource( mitk::IGIDataSource* source )
{
  m_Source = source;
  emit NewSourceAssociated(m_Source);
}

