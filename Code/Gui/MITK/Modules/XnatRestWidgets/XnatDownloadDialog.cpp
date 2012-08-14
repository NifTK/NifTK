#include <QtGui>
#include "XnatDownloadDialog.h"


XnatDownloadDialog::XnatDownloadDialog(QWidget* parent) : QDialog(parent), 
                                                          downloadCanceled(false)
{
    // delete dialog when dialog has accepted close event
    setAttribute(Qt::WA_DeleteOnClose);

    statusLabel = new QLabel(tr("Connecting..."));
    statusLabel->setAlignment(Qt::AlignCenter);

    QPushButton* cancelButton = new QPushButton(tr("Cancel Download"));
    connect(cancelButton, SIGNAL(clicked()), this, SLOT(cancelClicked()));

    QHBoxLayout* buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch();
    buttonLayout->addWidget(cancelButton);
    buttonLayout->addStretch();

    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(statusLabel);
    layout->addLayout(buttonLayout);
    setLayout(layout);

    setWindowTitle(tr("Download in Progress"));
    setFixedHeight(sizeHint().height());
    setMinimumWidth(300);
    setModal(true);
    setAttribute(Qt::WA_DeleteOnClose);
}

void XnatDownloadDialog::showBytesDownloaded(unsigned long numBytes)
{
    statusLabel->setText(tr("%1 bytes downloaded").arg(numBytes));
}

void XnatDownloadDialog::showUnzipInProgress()
{
    statusLabel->setText(tr("Unzipping downloaded file"));
    statusLabel->update();
}

void XnatDownloadDialog::cancelClicked()
{
    downloadCanceled = true;
}

void XnatDownloadDialog::closeEvent(QCloseEvent* event)
{
    if ( !downloadCanceled )
    {
        event->ignore();
        downloadCanceled = true;
    }
}

bool XnatDownloadDialog::close()
{
    downloadCanceled = true;
    return QWidget::close();
}
