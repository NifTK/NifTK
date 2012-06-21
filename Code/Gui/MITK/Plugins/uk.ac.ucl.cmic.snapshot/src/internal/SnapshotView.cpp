/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-11-30 05:43:06 +0000 (Wed, 30 Nov 2011) $
 Revision          : $Revision: 7891 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 
// Blueberry
#include "berryPlatform.h"
#include "berryISelectionService.h"
#include "berryIWorkbenchWindow.h"
#include "berryIWorkbenchPage.h"

// Qmitk
#include "SnapshotView.h"
#include "QmitkRenderWindow.h"

// MITK
#include "mitkIRenderWindowPart.h"

// Qt
#include <QMessageBox>
#include <QPixmap>
#include <QFileDialog>

// VTK
#include "vtkRenderWindow.h"
#include "vtkRenderLargeImage.h"
#include "vtkImageWriter.h"
#include "vtkPNGWriter.h"
#include "vtkJPEGWriter.h"

const std::string SnapshotView::VIEW_ID = "uk.ac.ucl.cmic.snapshot";

SnapshotView::SnapshotView()
: m_Controls(NULL)
, m_Parent(NULL)
{
}

SnapshotView::~SnapshotView()
{
}

std::string SnapshotView::GetViewID() const
{
  return VIEW_ID;
}

void SnapshotView::CreateQtPartControl( QWidget *parent )
{
  m_Parent = parent;

  if (!m_Controls)
  {
    // create GUI widgets from the Qt Designer's .ui file
    m_Controls = new Ui::SnapshotViewControls();
    m_Controls->setupUi( parent );

    // Connect Qt signals and slots programmatically.
    connect(m_Controls->m_TakeSnapshotButton, SIGNAL(pressed()), this, SLOT(OnTakeSnapshotButtonPressed()));
  }
}

void SnapshotView::SetFocus()
{
}

void SnapshotView::OnTakeSnapshotButtonPressed()
{
  int magnificationFactor = 1;
  QString windowName = tr("Snapshot");
  QString fileName = QFileDialog::getSaveFileName( m_Parent, tr("Save Snapshot As ..."), QDir::currentPath(), "PNG file (*.png);;JPEG file (*.jpg)" );

  mitk::IRenderWindowPart *renderWindowPart = this->GetRenderWindowPart();
  if (renderWindowPart != NULL)
  {
    QmitkRenderWindow *window = renderWindowPart->GetActiveRenderWindow();
    if (window != NULL)
    {
      vtkRenderer *renderer = window->GetRenderer()->GetVtkRenderer();
      if (renderer != NULL)
      {

        // Basically, "inspired by" QmitkSimpleExampleView.cpp

        bool doubleBuffering( renderer->GetRenderWindow()->GetDoubleBuffer() );

        renderer->GetRenderWindow()->DoubleBufferOff();

        vtkImageWriter* fileWriter;

        QFileInfo fi(fileName);
        QString suffix = fi.suffix();
        if (suffix.compare("png", Qt::CaseInsensitive) == 0)
        {
          fileWriter = vtkPNGWriter::New();
        }
        else  // default is jpeg
        {
          vtkJPEGWriter* w = vtkJPEGWriter::New();
          w->SetQuality(100);
          w->ProgressiveOff();
          fileWriter = w;
        }

        vtkRenderLargeImage* magnifier = vtkRenderLargeImage::New();
        magnifier->SetInput(renderer);
        magnifier->SetMagnification(magnificationFactor);

        fileWriter->SetInput(magnifier->GetOutput());
        fileWriter->SetFileName(fileName.toLatin1());
        fileWriter->Write();
        fileWriter->Delete();

        renderer->GetRenderWindow()->SetDoubleBuffer(doubleBuffering);

      }
      else
      {
        QMessageBox::critical(m_Parent, windowName,
            QString("Unknown ERROR: Failed to find VTK renderer. Please report this."),
            QMessageBox::Ok);
        return;
      }
    }
    else
    {
      QMessageBox::critical(m_Parent, windowName,
          QString("Unknown ERROR: Failed to find render window. Please report this."),
          QMessageBox::Ok);
      return;
    }
  }
  else
  {
    QMessageBox::critical(m_Parent, windowName,
        QString("Unknown ERROR: Failed to find render window part. Please report this."),
        QMessageBox::Ok);
    return;
  }
  return;
}
