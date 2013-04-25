/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkLaparoscopicSurgeryManager.h"
#include <mitkDataStorage.h>

//-----------------------------------------------------------------------------
QmitkLaparoscopicSurgeryManager::QmitkLaparoscopicSurgeryManager()
: m_DataStorage(NULL)
{
}


//-----------------------------------------------------------------------------
QmitkLaparoscopicSurgeryManager::~QmitkLaparoscopicSurgeryManager()
{
}


//-----------------------------------------------------------------------------
void QmitkLaparoscopicSurgeryManager::SetDataStorage(mitk::DataStorage* dataStorage)
{
  m_DataStorage = dataStorage;
  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkLaparoscopicSurgeryManager::setupUi(QWidget* parent)
{
  Ui_QmitkLaparoscopicSurgeryManager::setupUi(parent);
}


//-----------------------------------------------------------------------------
void QmitkLaparoscopicSurgeryManager::Update()
{
  std::cerr << "QmitkLaparoscopicSurgeryManager::Update - should do something interesting" << std::endl;
}
