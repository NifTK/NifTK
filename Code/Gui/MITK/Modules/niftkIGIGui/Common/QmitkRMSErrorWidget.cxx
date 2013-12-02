/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkRMSErrorWidget.h"
#include <mitkNodePredicateDataType.h>
#include <mitkPointSet.h>

//-----------------------------------------------------------------------------
QmitkRMSErrorWidget::QmitkRMSErrorWidget(QWidget *parent)
: m_DataStorage(NULL)
{
  setupUi(this);
}


//-----------------------------------------------------------------------------
QmitkRMSErrorWidget::~QmitkRMSErrorWidget()
{
}


//-----------------------------------------------------------------------------
void QmitkRMSErrorWidget::SetDataStorage(const mitk::DataStorage* dataStorage)
{
  m_DataStorage = const_cast<mitk::DataStorage*>(dataStorage);

  mitk::TNodePredicateDataType<mitk::PointSet>::Pointer isPointSet = mitk::TNodePredicateDataType<mitk::PointSet>::New();

  m_FixedCombo->SetDataStorage(m_DataStorage);
  m_FixedCombo->SetPredicate(isPointSet);
  m_FixedCombo->SetAutoSelectNewItems(false);

  m_MovingCombo->SetDataStorage(m_DataStorage);
  m_MovingCombo->SetPredicate(isPointSet);
  m_MovingCombo->SetAutoSelectNewItems(false);
}


//-----------------------------------------------------------------------------
double QmitkRMSErrorWidget::UpdateTransformation(const vtkMatrix4x4& matrix)
{
  return 0;      
}
