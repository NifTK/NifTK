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
#include <mitkCoordinateAxesData.h>
#include <vtkSmartPointer.h>
#include <mitkPointUtils.h>

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
  
  mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isTransform = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();
  m_TransformCombo->SetDataStorage(m_DataStorage);
  m_TransformCombo->SetPredicate(isTransform);
  m_TransformCombo->SetAutoSelectNewItems(false);
}


//-----------------------------------------------------------------------------
void QmitkRMSErrorWidget::Update()
{
  if (m_FixedCombo->GetSelectedNode().IsNull())
  {
    m_RMSError->setText("Fixed point set has not been specified.");
    return;
  }
  
  if (m_MovingCombo->GetSelectedNode().IsNull())
  {
    m_RMSError->setText("Moving point set has not been specified.");
    return;
  }
  
  if (m_TransformCombo->GetSelectedNode().IsNull())
  {
    m_RMSError->setText("Transform node has not been specified.");
    return;
  }
  
  mitk::PointSet::ConstPointer fixedPoints = dynamic_cast<mitk::PointSet*>(m_FixedCombo->GetSelectedNode().GetPointer());
  if (fixedPoints.IsNull())
  {
    m_RMSError->setText("Fixed point set is invalid.");
    return;
  }
  
  mitk::PointSet::ConstPointer movingPoints = dynamic_cast<mitk::PointSet*>(m_MovingCombo->GetSelectedNode().GetPointer());
  if (movingPoints.IsNull())
  {
    m_RMSError->setText("Moving point set is invalid.");
    return;
  }
  
  mitk::CoordinateAxesData::ConstPointer transform = dynamic_cast<mitk::CoordinateAxesData*>(m_TransformCombo->GetSelectedNode().GetPointer());
  if (transform.IsNull())
  {
    m_RMSError->setText("Transform is invalid.");
    return;    
  }
  
  if (movingPoints->GetSize() != fixedPoints->GetSize())
  {
    m_RMSError->setText("Fixed and moving points have different numbers of points.");
    return;
  }
  
  if (movingPoints->GetSize() == 0)
  {
    m_RMSError->setText("Moving point set is empty.");
    return;    
  }
  
  if (fixedPoints->GetSize() == 0)
  {
    m_RMSError->setText("Fixed point set is empty.");
    return;    
  }
  
  double rms = mitk::GetRMSErrorBetweenPoints(*fixedPoints, *movingPoints, transform.GetPointer());
  QString rmsString = QString::number(rms);
  m_RMSError->setText(rmsString);
}
