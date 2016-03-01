/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGITrackerSourceGui.h"
#include "QmitkIGITrackerSource.h"
#include "QmitkIGIDataSourceMacro.h"
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

NIFTK_IGISOURCE_GUI_MACRO(NIFTKIGIDATASOURCES_EXPORT, QmitkIGITrackerSourceGui, "IGI Tracker Source Gui")

//-----------------------------------------------------------------------------
QmitkIGITrackerSourceGui::QmitkIGITrackerSourceGui()
: m_TrackerSource(NULL)
{

}


//-----------------------------------------------------------------------------
QmitkIGITrackerSourceGui::~QmitkIGITrackerSourceGui()
{
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSourceGui::Initialize(QWidget* /* parent */, niftk::NiftyLinkClientDescriptor* /*config*/)
{
  setupUi(this);
  m_GroupBox->setCollapsed(true);

  if (this->GetSource() != NULL)
  {
    m_TrackerSource = dynamic_cast<QmitkIGITrackerSource*>(this->GetSource());

    if (m_TrackerSource != NULL)
    {
      vtkSmartPointer<vtkMatrix4x4> pre = m_TrackerSource->ClonePreMultiplyMatrix();
      m_PreTransformWidget->SetMatrix(*pre);

      vtkSmartPointer<vtkMatrix4x4> post = m_TrackerSource->ClonePostMultiplyMatrix();
      m_PostTransformWidget->SetMatrix(*post);
    }
  }

  connect(m_PreTransformWidget, SIGNAL(MatrixChanged()), this, SLOT(OnPreMatrixModified()));
  connect(m_PostTransformWidget, SIGNAL(MatrixChanged()), this, SLOT(OnPostMatrixModified()));
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSourceGui::OnPreMatrixModified()
{
  if (m_TrackerSource != NULL)
  {
    vtkSmartPointer<vtkMatrix4x4> pre = m_PreTransformWidget->CloneMatrix();
    m_TrackerSource->SetPreMultiplyMatrix(*pre);
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSourceGui::OnPostMatrixModified()
{
  if (m_TrackerSource != NULL)
  {
    vtkSmartPointer<vtkMatrix4x4> post = m_PostTransformWidget->CloneMatrix();
    m_TrackerSource->SetPostMultiplyMatrix(*post);
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSourceGui::Update()
{
  if (m_TrackerSource != NULL)
  {
    QString message = m_TrackerSource->GetStatusMessage();
    m_ConsoleDisplay->setPlainText(message);
    m_ConsoleDisplay->update();
  }
}
