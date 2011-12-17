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
#include "QmitkStdMultiWidgetEditor.h"
#include "QmitkStdMultiWidget.h"
#include "QmitkMIDASMultiViewEditor.h"
#include "QmitkMIDASMultiViewWidget.h"

// Qt
#include <QMessageBox>
#include <QPixmap>
#include <QFileDialog>

const std::string SnapshotView::VIEW_ID = "uk.ac.ucl.cmic.snapshot";

SnapshotView::SnapshotView()
: m_Parent(NULL)
{
}

SnapshotView::~SnapshotView()
{
}

void SnapshotView::CreateQtPartControl( QWidget *parent )
{
  m_Parent = parent;

  // create GUI widgets from the Qt Designer's .ui file
  m_Controls.setupUi( parent );
  connect(m_Controls.m_TakeSnapshotButton, SIGNAL(pressed()), this, SLOT(OnTakeSnapshotButtonPressed()));
}

void SnapshotView::OnTakeSnapshotButtonPressed()
{
  QWidget *widget = NULL;

  berry::IEditorPart::Pointer editor = this->GetSite()->GetPage()->GetActiveEditor();

  if (editor.Cast<QmitkStdMultiWidgetEditor>().IsNull() && editor.Cast<QmitkMIDASMultiViewEditor>().IsNull())
  {
    QMessageBox::warning(m_Parent,
        QString("Warning"),
        QString("Unfortunately we cannot take a screenshot of this type of editor"),
        QMessageBox::Ok);
    return;
  }

  if (editor.Cast<QmitkStdMultiWidgetEditor>().IsNotNull())
  {
    widget = editor.Cast<QmitkStdMultiWidgetEditor>()->GetStdMultiWidget();
  }
  else if (editor.Cast<QmitkMIDASMultiViewEditor>().IsNotNull())
  {
    widget = editor.Cast<QmitkMIDASMultiViewEditor>()->GetMIDASMultiViewWidget();
  }

  if (widget != NULL)
  {
    bool imageSaved = true;

    QPixmap snapshotImage = QPixmap::grabWindow(widget->winId());

    if(!snapshotImage.isNull())
    {

      QString fileName = QFileDialog::getSaveFileName( m_Parent, tr("Save Snapshot As"), "", tr("*.png;;*.bmp;;*.jpg;;*.jpeg;;All files(*.*)") );

      if(!fileName.isNull())
      {
        MITK_DEBUG << "SnapshotView::OnTakeSnapshotButtonPressed(): saving file to " << fileName.toLocal8Bit().constData() << std::endl;

        QStringList splitString = fileName.split(".");
        QString fileExtension;

        if (splitString.length() != 1)
        {
          // An extension has been specified.
          fileExtension = splitString[splitString.size() -1];
          if (fileExtension != "png"
              && fileExtension != "bmp"
              && fileExtension != "jpg"
              && fileExtension != "jpeg"
             )
          {
            QMessageBox::warning(m_Parent,
                QString("Warning"),
                QString("Invalid file extension, only .png, .bmp, .jpg and .jpeg are allowed"),
                QMessageBox::Ok);
            return;
          }
          else
          {
            std::string fileFormat = fileExtension.toLocal8Bit().constData();
            const char* chFormat = fileFormat.c_str();

            // Here is where you save the image
            imageSaved = snapshotImage.save(fileName, chFormat, -1);
          }
        }

        if(!imageSaved)
        {
          QMessageBox::critical(m_Parent, QString("CMIC Snapshot"),
              QString("Unknown ERROR: The Snapshot couldn't be saved. Please report this."),
              QMessageBox::Ok);
          return;
        }
      } // end if we have a filename
    } // end if we have successfully grabbed an image.
    else
    {
      QMessageBox::critical(m_Parent, QString("CMIC Snapshot"),
          QString("Unknown ERROR: Failed to grab an image. Please report this."),
          QMessageBox::Ok);
      return;
    }
  } // end if we have a widget
  else
  {
    QMessageBox::critical(m_Parent, tr("CMIC Snapshot"),
        QString("Unknown ERROR: Failed to grab an image. Please report this."),
        QMessageBox::Ok);
    return;
  }
}
