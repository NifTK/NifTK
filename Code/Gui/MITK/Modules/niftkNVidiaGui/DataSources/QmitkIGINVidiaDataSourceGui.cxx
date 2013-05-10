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
  : m_OglWin(0), m_PreviousBaseResolution(0)
{
  // To do.
}


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSourceGui::~QmitkIGINVidiaDataSourceGui()
{
  // gui is destroyed before data source (by igi data manager)
  // so disconnect ourselfs from source
  QmitkIGINVidiaDataSource* source = GetQmitkIGINVidiaDataSource();
  if (source)
  {
    // this is receiver
    // and source is sender
    this->disconnect(source);
  }

  // FIXME: not sure how to properly cleanup qt
  
  delete m_OglWin;
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

  QmitkIGINVidiaDataSource *source = this->GetQmitkIGINVidiaDataSource();
  if (source != NULL)
  {
    if (m_OglWin == 0)
    {
      // query for ogl context, etc
      // this should never fail, even if there's no sdi hardware
      QGLWidget* capturecontext = source->GetCaptureContext();
      assert(capturecontext != 0);

      // one preview window for all channels
      m_OglWin = new QmitkVideoPreviewWidget(PreviewGroupBox, capturecontext);
      PreviewGroupBox->layout()->addWidget(m_OglWin);
      m_OglWin->show();

      connect(source, SIGNAL(UpdateDisplay()), this, SLOT(OnUpdateDisplay()));

      // explicitly update at least once
      OnUpdateDisplay();
    }
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
    int streamcount = source->GetNumberOfStreams();
    std::ostringstream    sc;
    sc << streamcount;
    QString   ss = QString::fromStdString(sc.str());
    // only change text if it's actually different
    // otherwise the window is resetting a selection all the time: annoying as hell
    if (StreamCountTextBox->text().compare(ss) != 0)
    {
      StreamCountTextBox->setText(ss);
    }

    if (streamcount > 0)
    {
      int width = source->GetCaptureWidth();
      int height = source->GetCaptureHeight();
      float rr = source->GetRefreshRate();

      std::ostringstream    sf;
      sf << width << " x " << height << " @ " << rr << " Hz";

      ss = QString::fromStdString(sf.str());
      // only change text if it's actually different
      // otherwise the window is resetting a selection all the time: annoying as hell
      if (SignalTextBox->text().compare(ss) != 0)
      {
        SignalTextBox->setText(ss);
      }

      // there should be only one, really
      assert(PreviewGroupBox->layout() != 0);
      for (int i = 0; i < PreviewGroupBox->layout()->count(); ++i)
      {
        QLayoutItem* l = PreviewGroupBox->layout()->itemAt(i);
        QWidget*     w = l->widget();
        if (w)
        {
          QmitkVideoPreviewWidget*   g = dynamic_cast<QmitkVideoPreviewWidget*>(w);
          if (g)
          {
            g->SetVideoDimensions(width, height);
            g->SetTextureId(source->GetTextureId(0));
            g->updateGL();
            // one preview widget for all input streams
            break;
          }
        }
      }
    } // if streamcount
  }
}
