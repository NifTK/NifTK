#include "XnatUploadDialog.h"

#include <QCloseEvent>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>

XnatUploadDialog::XnatUploadDialog(QWidget* parent) : QDialog(parent), uploadCanceled(false)
{
  // delete dialog when dialog has accepted close event
  setAttribute(Qt::WA_DeleteOnClose);

  statusLabel = new QLabel(tr("Zipping files for upload..."));
  statusLabel->setAlignment(Qt::AlignCenter);

  QPushButton* cancelButton = new QPushButton(tr("Cancel Upload"));
  connect(cancelButton, SIGNAL(clicked()), this, SLOT(cancelClicked()));

  QHBoxLayout* buttonLayout = new QHBoxLayout;
  buttonLayout->addStretch();
  buttonLayout->addWidget(cancelButton);
  buttonLayout->addStretch();

  QVBoxLayout* layout = new QVBoxLayout;
  layout->addWidget(statusLabel);
  layout->addLayout(buttonLayout);
  setLayout(layout);

  setWindowTitle(tr("Upload in Progress"));
  setFixedHeight(sizeHint().height());
  setMinimumWidth(300);
  setModal(true);
  setAttribute(Qt::WA_DeleteOnClose);
}

bool XnatUploadDialog::wasUploadCanceled()
{
  return uploadCanceled;
}

void XnatUploadDialog::showUploadStarting()
{
  statusLabel->setText(tr("Connecting..."));
  // statusLabel->update();
}

void XnatUploadDialog::showBytesUploaded(unsigned long numBytes)
{
  statusLabel->setText(tr("%1 bytes uploaded").arg(numBytes));
}

void XnatUploadDialog::cancelClicked()
{
  uploadCanceled = true;
}

void XnatUploadDialog::closeEvent(QCloseEvent* event)
{
  if ( !uploadCanceled )
  {
    event->ignore();
    uploadCanceled = true;
  }
}

bool XnatUploadDialog::close()
{
  uploadCanceled = true;
  return QWidget::close();
}
