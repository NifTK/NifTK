/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkIGIOpenCVDataSourceGui.h"
#include <QImage>
#include <QPixmap>
#include <QLabel>
#include <QGridLayout>

#include <QmitkVideoBackground.h>
#include <QmitkRenderWindow.h>
#include "QmitkIGIDataSourceMacro.h"
#include "QmitkIGIOpenCVDataSource.h"
#include "QmitkStdMultiWidget.h"

NIFTK_IGISOURCE_GUI_MACRO(NIFTKIGIGUI_EXPORT, QmitkIGIOpenCVDataSourceGui, "IGI Open CV Video Gui")

//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSourceGui::QmitkIGIOpenCVDataSourceGui()
{
  // We run this class with its own RenderingManager so that you don't
  // get rendering updates causing re-rendering before the video framebuffer is ready.
  m_RenderingManager = mitk::RenderingManager::New();
}


//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSourceGui::~QmitkIGIOpenCVDataSourceGui()
{
  this->disconnect();
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
    source->Initialize(m_RenderWindow);
    connect(source, SIGNAL(UpdateDisplay()), this, SLOT(OnUpdateDisplay()));
  }
  else
  {
    MITK_ERROR << "QmitkIGIOpenCVDataSourceGui: source is NULL, which suggests a programming bug" << std::endl;
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIOpenCVDataSourceGui::OnUpdateDisplay()
{
  m_RenderingManager->RequestUpdateAll();
}
