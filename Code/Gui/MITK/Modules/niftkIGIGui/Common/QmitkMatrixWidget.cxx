/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMatrixWidget.h"
#include <QFileDialog>
#include <QString>
#include <QMessageBox>
#include <QDesktopServices>
#include <mitkFileIOUtils.h>

//-----------------------------------------------------------------------------
QmitkMatrixWidget::QmitkMatrixWidget(QWidget *parent)
{
  setupUi(this);
  m_Matrix = vtkMatrix4x4::New();
  m_Matrix->Identity();
  m_MatrixWidget->setEditable(false);
  m_MatrixWidget->setRange(-1e10, 1e10);
  this->connect(m_ClearButton, SIGNAL(pressed()), this, SLOT(OnClearButtonPressed()));
  this->connect(m_LoadButton, SIGNAL(pressed()), this, SLOT(OnLoadButtonPressed()));
}


//-----------------------------------------------------------------------------
QmitkMatrixWidget::~QmitkMatrixWidget()
{
}


//-----------------------------------------------------------------------------
void QmitkMatrixWidget::SetClearButtonVisible(const bool& isVisible)
{
  m_ClearButton->setVisible(isVisible);
}


//-----------------------------------------------------------------------------
void QmitkMatrixWidget::SetLoadButtonVisible(const bool& isVisible)
{
  m_LoadButton->setVisible(isVisible);
}


//-----------------------------------------------------------------------------
void QmitkMatrixWidget::SynchroniseWidgetWithMatrix()
{
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      m_MatrixWidget->setValue(i, j, m_Matrix->GetElement(i, j));
    }
  }
  emit MatrixChanged();
}


//-----------------------------------------------------------------------------
void QmitkMatrixWidget::OnClearButtonPressed()
{
  m_Matrix->Identity();
  this->SynchroniseWidgetWithMatrix();
}


//-----------------------------------------------------------------------------
void QmitkMatrixWidget::OnLoadButtonPressed()
{
  QString fileName = QFileDialog::getOpenFileName(this,
                       tr("Load Matrix"),
                       QDesktopServices::storageLocation(QDesktopServices::HomeLocation),
                       tr("Matrix Files (*.txt *.mat *.4x4)")
                     );

  if (fileName.size() > 0)
  {
    // TO DO: Better error handling
    m_Matrix = mitk::LoadVtkMatrix4x4FromFile(fileName.toStdString());
    this->SynchroniseWidgetWithMatrix();
  }
}


//-----------------------------------------------------------------------------
void QmitkMatrixWidget::SetMatrix(const vtkMatrix4x4& matrix)
{
  m_Matrix->DeepCopy(&matrix);
  this->SynchroniseWidgetWithMatrix();
}


//-----------------------------------------------------------------------------
vtkMatrix4x4* QmitkMatrixWidget::CloneMatrix() const
{
  vtkMatrix4x4* tmp = vtkMatrix4x4::New();
  tmp->DeepCopy(this->m_Matrix);
  return tmp;
}

