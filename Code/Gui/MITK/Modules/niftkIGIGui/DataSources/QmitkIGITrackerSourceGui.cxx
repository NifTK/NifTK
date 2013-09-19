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
#include <Common/NiftyLinkXMLBuilder.h>
#include "QmitkIGIDataSourceMacro.h"

NIFTK_IGISOURCE_GUI_MACRO(NIFTKIGIGUI_EXPORT, QmitkIGITrackerSourceGui, "IGI Tracker Source Gui")

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
void QmitkIGITrackerSourceGui::Initialize(QWidget* /* parent */, ClientDescriptorXMLBuilder* /*config*/)
{
  setupUi(this);

  if (this->GetSource() != NULL)
  {
    m_TrackerSource = dynamic_cast<QmitkIGITrackerSource*>(this->GetSource());
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
