/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGINVidiaDataSourceGui.h"
#include <QImage>
#include <QPixmap>
#include <QLabel>
#include <QGridLayout>
#include <QGLWidget>
#include "QmitkIGIDataSourceMacro.h"
#include "QmitkIGINVidiaDataSource.h"
#include "QmitkVideoPreviewWidget.h"

NIFTK_IGISOURCE_GUI_MACRO(NIFTKNVIDIAGUI_EXPORT, QmitkIGINVidiaDataSourceGui, "IGI NVidia Video Gui")

//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSourceGui::QmitkIGINVidiaDataSourceGui()
: m_DisplayWidget(NULL)
, m_Layout(NULL)
{
  // To do.
}


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSourceGui::~QmitkIGINVidiaDataSourceGui()
{
  // To do.
}


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSource* QmitkIGINVidiaDataSourceGui::GetQmitkIGINVidiaDataSource() const
{
  QmitkIGINVidiaDataSource* result = NULL;

  if (this->GetSource() != NULL)
  {
    result = dynamic_cast<QmitkIGINVidiaDataSource*>(this->GetSource());
  }

  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceGui::Initialize(QWidget *parent)
{
  setupUi(this);
  /*
  this->setParent(parent);

  m_Layout = new QGridLayout(this);
  m_Layout->setContentsMargins(0,0,0,0);
  m_Layout->setSpacing(0);

  m_DisplayWidget = new QLabel(this);
  m_DisplayWidget->setText("Update me, to show some kind of preview picture");
  m_Layout->addWidget(m_DisplayWidget, 0, 0);
  */

  QmitkIGINVidiaDataSource *source = this->GetQmitkIGINVidiaDataSource();
  if (source != NULL)
  {
    // query for ogl context, etc
    // this should never fail, even if there's no sdi hardware
    QGLWidget* capturecontext = source->get_capturecontext();
    assert(capturecontext != 0);

    // FIXME: one for each channel
    QmitkVideoPreviewWidget* oglwin = new QmitkVideoPreviewWidget(this, capturecontext);
    previewgridlayout->addWidget(oglwin);
    oglwin->show();

    connect(source, SIGNAL(UpdateDisplay()), this, SLOT(OnUpdateDisplay()));
  }
  else
  {
    MITK_ERROR << "QmitkIGINVidiaDataSourceGui: source is NULL, which suggests a programming bug" << std::endl;
  }

}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceGui::OnUpdateDisplay()
{
  QmitkIGINVidiaDataSource *source = this->GetQmitkIGINVidiaDataSource();
  if (source != NULL)
  {
    int width = source->get_capture_width();
    int height = source->get_capture_height();
    float rr = source->get_refresh_rate();

    std::ostringstream    s;
    s << width << " x " << height << " @ " << rr << " Hz";

    QString   ss = QString::fromStdString(s.str());
    // only change text if it's actually different
    // otherwise the window is resetting a selection all the time: annoying as hell
    if (signal_tb->text().compare(ss) != 0)
      signal_tb->setText(ss);

    actualcaptureformat_tb->setText(QString::fromAscii("FIXME"));


    for (int i = 0; i < previewgridlayout->count(); ++i)
    {
      QLayoutItem* l = previewgridlayout->itemAt(i);
      QWidget*     w = l->widget();
      if (w)
      {
        QmitkVideoPreviewWidget*   g = dynamic_cast<QmitkVideoPreviewWidget*>(w);
        if (g)
        {
          g->set_texture_id(source->get_texture_id(0));
          g->updateGL();
        }
      }
    }
  }
}
