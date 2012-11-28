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

#include "QmitkIGIDataSourceManager.h"
#include <QMessageBox>
#include <QmitkStdMultiWidget.h>
#include <mitkDataStorage.h>
#include "mitkIGIDataSource.h"
#include "QmitkIGINiftyLinkDataSource.h"
#include "QmitkIGITrackerTool.h"

//-----------------------------------------------------------------------------
QmitkIGIDataSourceManager::QmitkIGIDataSourceManager()
: m_DataStorage(NULL)
, m_StdMultiWidget(NULL)
, m_GridLayoutClientControls(NULL)
, m_UpdateTimer(NULL)
, m_NextSourceIdentifier(0)
{
}


//-----------------------------------------------------------------------------
QmitkIGIDataSourceManager::~QmitkIGIDataSourceManager()
{
  if (m_UpdateTimer != NULL)
  {
    m_UpdateTimer->stop();
    delete m_UpdateTimer;
  }

  m_Sources.clear(); // smart pointers should delete the sources.
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnUpdateTimeOut()
{
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::SetDataStorage(mitk::DataStorage* dataStorage)
{
  m_DataStorage = dataStorage;

  foreach (mitk::IGIDataSource *source, m_Sources)
  {
    source->SetDataStorage(dataStorage);
  }
  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::setupUi(QWidget* parent)
{
  Ui_QmitkIGIDataSourceManager::setupUi(parent);

  m_UpdateTimer =  new QTimer(this);
  m_UpdateTimer->setInterval ( 50 );

  m_GridLayoutClientControls = new QGridLayout(m_Frame);
  m_GridLayoutClientControls->setSpacing(0);
  m_GridLayoutClientControls->setContentsMargins(0, 0, 0, 0);

  m_Frame->setContentsMargins(0, 0, 0, 0);

  m_SourceSelectComboBox->addItem("tracker");
  m_SourceSelectComboBox->addItem("ultrasonix");
  m_SourceSelectComboBox->addItem("local framegrabber");

  connect(m_SourceSelectComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnCurrentIndexChanged(int)));
  connect(m_AddSourcePushButton, SIGNAL(clicked()), this, SLOT(OnAddSource()) );
  connect(m_RemoveSourcePushButton, SIGNAL(clicked()), this, SLOT(OnRemoveSource()) );
  connect(m_TableWidget, SIGNAL(cellDoubleClicked(int, int)), this, SLOT(OnCellDoubleClicked(int, int)) );
  connect(m_UpdateTimer, SIGNAL(timeout()), this, SLOT(OnUpdateTimeOut()) );

  m_SourceSelectComboBox->setCurrentIndex(0);
  m_UpdateTimer->start();
}


//-----------------------------------------------------------------------------
bool QmitkIGIDataSourceManager::IsPortSpecificType()
{
  bool isPortSpecific = false;
  int currentIndex = m_SourceSelectComboBox->currentIndex();

  if (currentIndex == 0 || currentIndex == 1)
  {
    isPortSpecific = true;
  }

  return isPortSpecific;
}


//-----------------------------------------------------------------------------
int QmitkIGIDataSourceManager::GetRowNumberFromIdentifier(int identifier)
{
  int rowNumber = -1;
  for (unsigned int i = 0; i < m_Sources.size(); i++)
  {
    if (m_Sources[i].IsNotNull() && m_Sources[i]->GetIdentifier() == identifier)
    {
      rowNumber = i;
      break;
    }
  }
  return rowNumber;
}


//-----------------------------------------------------------------------------
int QmitkIGIDataSourceManager::GetIdentifierFromRowNumber(int rowNumber)
{
  int identifier = -1;
  if (rowNumber < m_TableWidget->rowCount() && rowNumber < (int)m_Sources.size())
  {
    identifier = m_Sources[rowNumber]->GetIdentifier();
  }
  return identifier;
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::UpdateToolDisplay(int toolIdentifier)
{
  int rowNumber = this->GetRowNumberFromIdentifier(toolIdentifier);
  std::string status = m_Sources[toolIdentifier]->GetStatus();
  std::string type = m_Sources[toolIdentifier]->GetType();
  std::string device = m_Sources[toolIdentifier]->GetName();
  std::string description = m_Sources[toolIdentifier]->GetDescription();

  std::vector<std::string> fields;
  fields.push_back(status);
  fields.push_back(type);
  fields.push_back(device);
  fields.push_back(description);

  if (rowNumber != -1)
  {
    if (rowNumber == m_TableWidget->rowCount())
    {
      m_TableWidget->insertRow(rowNumber);
    }

    for (unsigned int i = 0; i < fields.size(); i++)
    {
      QTableWidgetItem *item = new QTableWidgetItem(QString::fromStdString(fields[i]));
      item->setTextAlignment(Qt::AlignCenter);
      item->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

      m_TableWidget->setItem(rowNumber, i, item);
    }

    m_TableWidget->show();
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnCurrentIndexChanged(int indexNumber)
{
  Q_UNUSED(indexNumber);

  if (this->IsPortSpecificType())
  {
    m_PortNumberSpinBox->setEnabled(true);
  }
  else
  {
    m_PortNumberSpinBox->setEnabled(false);
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnAddSource()
{
  int sourceType = m_SourceSelectComboBox->currentIndex();
  int portNumber = m_PortNumberSpinBox->value();

  mitk::IGIDataSource::Pointer source = NULL;

  if (this->IsPortSpecificType())
  {
    if (m_PortsInUse.contains(portNumber))
    {
      QMessageBox msgBox(QMessageBox::Warning, tr("Server failure"), tr("Could not open socket: already listening on the selected port!"), QMessageBox::Ok);
      msgBox.exec();
      return;
    }
  }

  if (sourceType == 0 || sourceType == 1)
  {
    QmitkIGINiftyLinkDataSource::Pointer niftyLinkSource = NULL;
    if (sourceType == 0)
    {
      niftyLinkSource = QmitkIGITrackerTool::New();
    }
    else if (sourceType == 1)
    {

    }
    if (niftyLinkSource->ListenOnPort(portNumber))
    {
      m_PortsInUse.insert(portNumber);
    }
    source = niftyLinkSource;
  }
  else if (sourceType == 1)
  {
    // create ultrasonix source
  }
  else if (sourceType == 2)
  {
    // create OpenCV local frame grabber.
  }
  else
  {
    std::cerr << "Matt, not implemented yet" << std::endl;
  }

  source->SetIdentifier(m_NextSourceIdentifier++);
  source->SetDataStorage(m_DataStorage);

  m_Sources.push_back(source);

  // Registers this class as a listener to any status updates and connects to UpdateToolDisplay.
  // This means that regardless of the tool type, this class will receive a signal, and then
  // callback to the tool to ask for the necessary data to update the GUI row.
  source->DataSourceStatusUpdated
    += mitk::MessageDelegate1<QmitkIGIDataSourceManager, int>(
        this, &QmitkIGIDataSourceManager::UpdateToolDisplay );

  // Force an update.
  source->DataSourceStatusUpdated.Send(source->GetIdentifier());
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnRemoveSource()
{
  if (m_TableWidget->rowCount() == 0)
    return;

  int rowIndex = m_TableWidget->currentRow();

  if (rowIndex < 0)
    rowIndex = m_TableWidget->rowCount()-1;


  mitk::IGIDataSource::Pointer source = m_Sources[rowIndex];

  // De-registers this class as a listener to this source.
  source->DataSourceStatusUpdated
    -= mitk::MessageDelegate1<QmitkIGIDataSourceManager, int>(
        this, &QmitkIGIDataSourceManager::UpdateToolDisplay );

  // If it is a networked tool, removes the port number from our list of "ports in use".
  if (source->GetNameOfClass() == std::string("QmitkIGITrackerTool").c_str()
    || source->GetNameOfClass() == std::string("QmitkIGIUltrasonixTool").c_str()
    )
  {
    QmitkIGINiftyLinkDataSource::Pointer niftyLinkSource = dynamic_cast<QmitkIGINiftyLinkDataSource*>(source.GetPointer());
    if (niftyLinkSource.IsNotNull())
    {
      int portNumber = niftyLinkSource->GetPort();
      m_PortsInUse.remove(portNumber);
    }
  }

  m_TableWidget->removeRow(rowIndex);

  // This destroys the source. It is up to the source to correctly destroy itself,
  // as this class has no idea what the source is or what it contains etc.
  m_Sources.erase(m_Sources.begin() + rowIndex);
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnCellDoubleClicked(int, int)
{
  // ToDo: Generate a GUI.
}
