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
{

}


//-----------------------------------------------------------------------------
QmitkIGITrackerSourceGui::~QmitkIGITrackerSourceGui()
{
}


//-----------------------------------------------------------------------------
QmitkIGITrackerSource* QmitkIGITrackerSourceGui::GetQmitkIGITrackerSource() const
{
  QmitkIGITrackerSource* result = NULL;
  if (this->GetSource() != NULL)
  {
    result = dynamic_cast<QmitkIGITrackerSource*>(this->GetSource());
  }

  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSourceGui::Initialize(QWidget *parent, ClientDescriptorXMLBuilder* config)
{
  setupUi(this);

  if (config != NULL)
  {
    TrackerClientDescriptor *trDesc = dynamic_cast<TrackerClientDescriptor*>(config);
    if (trDesc != NULL)
    {
      QmitkIGITrackerSource *source = this->GetQmitkIGITrackerSource();
      if (source != NULL)
      {
        connect (source, SIGNAL(StatusUpdate(QString)), this, SLOT(OnStatusUpdate(QString)));
      }
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSourceGui::OnStatusUpdate(QString message)
{
  m_ConsoleDisplay->setPlainText(message);
  m_ConsoleDisplay->update();
}
