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
#include "QmitkIGIDataSourceGui.h"
#include "QmitkIGINiftyLinkDataSource.h"
#include "QmitkIGITrackerTool.h"
#include "QmitkIGIUltrasonixTool.h"

//-----------------------------------------------------------------------------
QmitkIGIDataSourceManager::QmitkIGIDataSourceManager()
: m_DataStorage(NULL)
, m_StdMultiWidget(NULL)
, m_GridLayoutClientControls(NULL)
, m_UpdateTimer(NULL)
, m_FrameRateTimer(NULL)
, m_NextSourceIdentifier(0)
, m_FrameRate(2)
, m_DirectoryPrefix("")
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

  if (m_FrameRateTimer != NULL)
  {
    m_FrameRateTimer->stop();
    delete m_FrameRateTimer;
  }

  m_Sources.clear(); // smart pointers should delete the sources.
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
void QmitkIGIDataSourceManager::SetFramesPerSecond(int framesPerSecond)
{
  m_FrameRate = framesPerSecond;

  if (m_UpdateTimer != NULL)
  {
    int milliseconds = 1000 / framesPerSecond;
    m_FrameRateTimer->setInterval(milliseconds);
  }

  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::SetDirectoryPrefix(QString& directoryPrefix)
{
  m_DirectoryPrefix = directoryPrefix;
  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::SetErrorColour(QColor &colour)
{
  m_ErrorColour = colour;
  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::SetWarningColour(QColor &colour)
{
  m_WarningColour = colour;
  this->Modified();
}


void QmitkIGIDataSourceManager::SetOKColour(QColor &colour)
{
  m_OKColour = colour;
  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::setupUi(QWidget* parent)
{
  Ui_QmitkIGIDataSourceManager::setupUi(parent);

  m_PlayPushButton->setIcon(QIcon(":/niftkIGIGuiResources/play.png"));
  m_RecordPushButton->setIcon(QIcon(":/niftkIGIGuiResources/record.png"));
  m_StopPushButton->setIcon(QIcon(":/niftkIGIGuiResources/stop.png"));

  m_PlayPushButton->setEnabled(false); // not ready yet.

  m_UpdateTimer =  new QTimer(this);
  m_UpdateTimer->setInterval ( 500 ); // 2 times per second

  m_FrameRateTimer = new QTimer(this);
  m_FrameRateTimer->setInterval(1000); // every 1 seconds

  m_GridLayoutClientControls = new QGridLayout(m_Frame);
  m_GridLayoutClientControls->setSpacing(0);
  m_GridLayoutClientControls->setContentsMargins(0, 0, 0, 0);

  m_Frame->setContentsMargins(0, 0, 0, 0);

  m_SourceSelectComboBox->addItem("networked tracker");
  m_SourceSelectComboBox->addItem("networked ultrasonix scanner");
  m_SourceSelectComboBox->addItem("local frame grabber");

  connect(m_SourceSelectComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnCurrentIndexChanged(int)));
  connect(m_AddSourcePushButton, SIGNAL(clicked()), this, SLOT(OnAddSource()) );
  connect(m_RemoveSourcePushButton, SIGNAL(clicked()), this, SLOT(OnRemoveSource()) );
  connect(m_TableWidget, SIGNAL(cellDoubleClicked(int, int)), this, SLOT(OnCellDoubleClicked(int, int)) );
  connect(m_UpdateTimer, SIGNAL(timeout()), this, SLOT(OnUpdateDisplay()) );
  connect(m_FrameRateTimer, SIGNAL(timeout()), this, SLOT(OnUpdateFrameRate()) );

  m_SourceSelectComboBox->setCurrentIndex(0);
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
      niftyLinkSource = QmitkIGIUltrasonixTool::New();
    }
    if (niftyLinkSource->ListenOnPort(portNumber))
    {
      m_PortsInUse.insert(portNumber);
      m_PortNumberSpinBox->setValue(portNumber+1);
    }
    source = niftyLinkSource;
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

  // Launch timers
  if (!m_UpdateTimer->isActive())
  {
    m_UpdateTimer->start();
  }
  if (!m_FrameRateTimer->isActive())
  {
    m_FrameRateTimer->start();
  }
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

  if (m_TableWidget->rowCount() == 0)
  {
    m_UpdateTimer->stop();
    m_FrameRateTimer->stop();
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnCellDoubleClicked(int row, int column)
{
  QmitkIGIDataSourceGui* sourceGui = NULL;

  int identifier = this->GetIdentifierFromRowNumber(row);
  mitk::IGIDataSource* source = m_Sources[identifier];
  const std::string classname = source->GetNameOfClass();

  std::string guiClassname = classname + "Gui";

  std::list<itk::LightObject::Pointer> allGUIs = itk::ObjectFactoryBase::CreateAllInstance(guiClassname.c_str());
  for( std::list<itk::LightObject::Pointer>::iterator iter = allGUIs.begin();
       iter != allGUIs.end();
       ++iter )
  {
    if (sourceGui == NULL)
    {
      itk::LightObject* guiObject = (*iter).GetPointer();
      sourceGui = dynamic_cast<QmitkIGIDataSourceGui*>( guiObject );
    }
    else
    {
      MITK_ERROR << "There is more than one GUI for " << classname << " (several factories claim ability to produce a " << guiClassname << " ) " << std::endl;
      return;
    }
  }

  if (sourceGui != NULL)
  {
    if (m_GridLayoutClientControls != NULL)
    {
      delete m_GridLayoutClientControls;
    }

    m_GridLayoutClientControls = new QGridLayout(m_Frame);
    m_GridLayoutClientControls->setSpacing(0);
    m_GridLayoutClientControls->setContentsMargins(0, 0, 0, 0);

    sourceGui->SetStdMultiWidget(this->GetStdMultiWidget());
    sourceGui->SetDataSource(source);
    sourceGui->Initialize(NULL);

    m_GridLayoutClientControls->addWidget(sourceGui, 0, 0);
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnUpdateDisplay()
{
  igtl::TimeStamp::Pointer timeNow = igtl::TimeStamp::New();
  timeNow->toTAI();

  igtlUint64 idNow = timeNow->GetTimeStampUint64();

  foreach ( mitk::IGIDataSource::Pointer source, m_Sources )
  {
    // This is the main callback method to tell the source to update.
    // Each source will then inform its own GUI (if present) to update.
    bool isValid = source->ProcessData(idNow);

    int rowNumber = this->GetRowNumberFromIdentifier(source->GetIdentifier());
    QTableWidgetItem *tItem = m_TableWidget->item(rowNumber, 0);

    if (!isValid)
    {
      // Highlight that current row is in error.
      QPixmap pix(22, 22);
      pix.fill(QColor(Qt::red));
      tItem->setIcon(pix);
    }
    else
    {
      // Highlight that current row is OK.
      QPixmap pix(22, 22);
      pix.fill(QColor(Qt::green));
      tItem->setIcon(pix);
    }
  }

  mitk::RenderingManager * renderer = mitk::RenderingManager::GetInstance();
  renderer->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnUpdateFrameRate()
{
  foreach ( mitk::IGIDataSource::Pointer source, m_Sources )
  {
    source->UpdateFrameRate();
    float rate = source->GetFrameRate();

    double lag = source->GetCurrentTimeLag();

    int rowNumber = this->GetRowNumberFromIdentifier(source->GetIdentifier());

    QTableWidgetItem *frameRateItem = new QTableWidgetItem(QString::number(rate));
    frameRateItem->setTextAlignment(Qt::AlignCenter);
    frameRateItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    m_TableWidget->setItem(rowNumber, 4, frameRateItem);

    QTableWidgetItem *lagItem = new QTableWidgetItem(QString::number(lag));
    lagItem->setTextAlignment(Qt::AlignCenter);
    lagItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    m_TableWidget->setItem(rowNumber, 5, lagItem);
  }
}

