/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIOpenCVDataSourceGui.h"
#include <QImage>
#include <QPixmap>
#include <QLabel>
#include <QGridLayout>
#include <QmitkVideoBackground.h>
#include <QmitkRenderWindow.h>
#include "QmitkIGIDataSourceMacro.h"
#include "QmitkIGIOpenCVDataSource.h"
#include <QmitkStdMultiWidget.h>

NIFTK_IGISOURCE_GUI_MACRO(NIFTKIGIGUI_EXPORT, QmitkIGIOpenCVDataSourceGui, "IGI Open CV Video Gui")

//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSourceGui::QmitkIGIOpenCVDataSourceGui()
: m_Background(NULL)
, m_RenderWindow(NULL)
, m_RenderingManager(NULL)
, m_Layout(NULL)
{
  // We run this class with its own RenderingManager so that you don't
  // get rendering updates causing re-rendering before the video framebuffer is ready.
  m_RenderingManager = mitk::RenderingManager::New();
}


//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSourceGui::~QmitkIGIOpenCVDataSourceGui()
{
  // FIXME: what exactly are these disconnecting?
  m_Background->disconnect();
  this->disconnect();

  // gui is destroyed before data source (by igi data manager)
  // so disconnect ourselfs from source
  QmitkIGIOpenCVDataSource* source = GetOpenCVDataSource();
  if (source)
  {
    // this is receiver
    // and source is sender
    this->disconnect(source);
  }

  // delete render window first.
  if (m_RenderWindow != NULL)
  {
    delete m_RenderWindow;
  }
  if (m_Background != NULL)
  {
    delete m_Background;
  }
}


//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSource* QmitkIGIOpenCVDataSourceGui::GetOpenCVDataSource() const
{
  QmitkIGIOpenCVDataSource* result = NULL;

  if (this->GetSource() != NULL)
  {
    result = dynamic_cast<QmitkIGIOpenCVDataSource*>(this->GetSource());
  }

  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGIOpenCVDataSourceGui::Initialize(QWidget *parent)
{
  this->setParent(parent);

  m_Layout = new QGridLayout(this);
  m_Layout->setContentsMargins(0,0,0,0);
  m_Layout->setSpacing(0);

  m_RenderWindow = new QmitkRenderWindow(this, QString("QmitkIGIOpenCVDataSourceGui"), NULL, m_RenderingManager);
  m_Layout->addWidget(m_RenderWindow, 0, 0);

  QmitkIGIOpenCVDataSource *source = this->GetOpenCVDataSource();
  if (source != NULL)
  {
    m_Background = new QmitkVideoBackground(source->GetVideoSource());
    m_Background->AddRenderWindow(m_RenderWindow->GetVtkRenderWindow());
    m_Background->UpdateVideo();

    connect(source, SIGNAL(UpdateDisplay()), this, SLOT(OnUpdateDisplay()), Qt::QueuedConnection);
  }
  else
  {
    MITK_ERROR << "QmitkIGIOpenCVDataSourceGui: source is NULL, which suggests a programming bug" << std::endl;
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIOpenCVDataSourceGui::OnUpdateDisplay()
{
  m_Background->UpdateVideo();
  m_RenderingManager->RequestUpdateAll();
}
