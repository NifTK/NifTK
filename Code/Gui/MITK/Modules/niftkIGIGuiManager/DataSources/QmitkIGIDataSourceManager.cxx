/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIDataSourceManager.h"
#include <QMessageBox>
#include <QmitkStdMultiWidget.h>
#include <QDesktopServices>
#include <QDateTime>
#include <mitkDataStorage.h>
#include <mitkIGIDataSource.h>
#include <QmitkIGINiftyLinkDataSource.h>
#include <QmitkIGITrackerSource.h>
#include <QmitkIGIUltrasonixTool.h>
#include <QmitkIGIOpenCVDataSource.h>
#include <QmitkIGIDataSourceGui.h>

#ifdef _USE_NVAPI
#include <QmitkIGINVidiaDataSource.h>
#endif

const QColor QmitkIGIDataSourceManager::DEFAULT_ERROR_COLOUR = QColor(Qt::red);
const QColor QmitkIGIDataSourceManager::DEFAULT_WARNING_COLOUR = QColor(255,127,0); // orange
const QColor QmitkIGIDataSourceManager::DEFAULT_OK_COLOUR = QColor(Qt::green);
const QColor QmitkIGIDataSourceManager::DEFAULT_SUSPENDED_COLOUR = QColor(Qt::blue);
const int    QmitkIGIDataSourceManager::DEFAULT_FRAME_RATE = 2; // twice per second
const int    QmitkIGIDataSourceManager::DEFAULT_CLEAR_RATE = 2; // every 2 seconds
const int    QmitkIGIDataSourceManager::DEFAULT_TIMING_TOLERANCE = 5000; // 5 seconds expressed in milliseconds
const bool   QmitkIGIDataSourceManager::DEFAULT_SAVE_ON_RECEIPT = true;
const bool   QmitkIGIDataSourceManager::DEFAULT_SAVE_IN_BACKGROUND = false;

//-----------------------------------------------------------------------------
QmitkIGIDataSourceManager::QmitkIGIDataSourceManager()
: m_DataStorage(NULL)
, m_StdMultiWidget(NULL)
, m_GridLayoutClientControls(NULL)
, m_NextSourceIdentifier(0)
, m_GuiUpdateTimer(NULL)
, m_ClearDownTimer(NULL)
, m_PlaybackSliderBase(0)
, m_PlaybackSliderFactor(1)
{
  m_SuspendedColour = DEFAULT_SUSPENDED_COLOUR;
  m_OKColour = DEFAULT_OK_COLOUR;
  m_WarningColour = DEFAULT_WARNING_COLOUR;
  m_ErrorColour = DEFAULT_ERROR_COLOUR;
  m_FrameRate = DEFAULT_FRAME_RATE;
  m_ClearDataRate = DEFAULT_CLEAR_RATE;
  m_TimingTolerance = DEFAULT_TIMING_TOLERANCE;
  m_DirectoryPrefix = GetDefaultPath();
  m_SaveOnReceipt = DEFAULT_SAVE_ON_RECEIPT;
  m_SaveInBackground = DEFAULT_SAVE_IN_BACKGROUND;
}


//-----------------------------------------------------------------------------
QmitkIGIDataSourceManager::~QmitkIGIDataSourceManager()
{
  // Stop both timers, to make sure nothing is triggering as we destroy.
  if (m_GuiUpdateTimer != NULL)
  {
    m_GuiUpdateTimer->stop();
  }

  if (m_ClearDownTimer != NULL)
  {
    m_ClearDownTimer->stop();
  }

  this->DeleteCurrentGuiWidget();
  m_Sources.clear();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::DeleteCurrentGuiWidget()
{
  if (m_GridLayoutClientControls != NULL)
  {
    QLayoutItem *item = m_GridLayoutClientControls->itemAtPosition(0,0);
    if (item != NULL)
    {
      QWidget *widget = item->widget();
      if (widget != NULL)
      {
        widget->setVisible(false);
      }
      m_GridLayoutClientControls->removeItem(item);
      // this gets rid of the layout item
      delete item;
      // we should kill off the actual widget too
      // side note: that may not be the best way of doing it. currently everytime you
      //  double-click on a source it creates a new gui for it and deletes the previously
      //  active gui. it would be better if guis are cached for as long as the underlying
      //  data source is alive.
      // but for now this.
      delete widget;
    }
    delete m_GridLayoutClientControls;
    m_GridLayoutClientControls = 0;
  }
}


//-----------------------------------------------------------------------------
QString QmitkIGIDataSourceManager::GetDefaultPath()
{
  QString path;
  QDir directory;

  path = QDesktopServices::storageLocation(QDesktopServices::DesktopLocation);
  directory.setPath(path);

  if (!directory.exists())
  {
    path = QDesktopServices::storageLocation(QDesktopServices::DocumentsLocation);
    directory.setPath(path);
  }
  if (!directory.exists())
  {
    path = QDesktopServices::storageLocation(QDesktopServices::HomeLocation);
    directory.setPath(path);
  }
  if (!directory.exists())
  {
    path = QDir::currentPath();
    directory.setPath(path);
  }
  if (!directory.exists())
  {
    path = "";
  }
  return path;
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::SetDataStorage(mitk::DataStorage* dataStorage)
{
  m_DataStorage = dataStorage;

  foreach (QmitkIGIDataSource *source, m_Sources)
  {
    source->SetDataStorage(dataStorage);
  }

  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::SetFramesPerSecond(const int& framesPerSecond)
{
  if (m_GuiUpdateTimer != NULL)
  {
    int milliseconds = 1000 / framesPerSecond;
    m_GuiUpdateTimer->setInterval(milliseconds);
  }

  m_FrameRate = framesPerSecond;
  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::SetClearDataRate(const int& numberOfSeconds)
{
  if (m_ClearDownTimer != NULL)
  {
    int milliseconds = 1000 * numberOfSeconds;
    m_ClearDownTimer->setInterval(milliseconds);
  }

  m_ClearDataRate = numberOfSeconds;
  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::SetTimingTolerance(const int& timingTolerance)
{
  m_TimingTolerance = (igtlUint64)timingTolerance*(igtlUint64)1000000; // input is in milliseconds, but the underlying source is in nano-seconds.
  for (unsigned int i = 0; i < m_Sources.size(); i++)
  {
    m_Sources[i]->SetTimeStampTolerance(m_TimingTolerance);
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


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::SetOKColour(QColor &colour)
{
  m_OKColour = colour;
  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::SetSuspendedColour(QColor &colour)
{
  m_SuspendedColour = colour;
  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::setupUi(QWidget* parent)
{
  Ui_QmitkIGIDataSourceManager::setupUi(parent);

  m_PlayPushButton->setIcon(QIcon(":/niftkIGIGuiManagerResources/play.png"));
  m_RecordPushButton->setIcon(QIcon(":/niftkIGIGuiManagerResources/record.png"));
  m_StopPushButton->setIcon(QIcon(":/niftkIGIGuiManagerResources/stop.png"));

  m_PlayPushButton->setEnabled(true);
  m_RecordPushButton->setEnabled(true);
  m_StopPushButton->setEnabled(false);

  m_GuiUpdateTimer = new QTimer(this);
  m_GuiUpdateTimer->setInterval(1000/(int)(DEFAULT_FRAME_RATE));

  m_ClearDownTimer = new QTimer(this);
  m_ClearDownTimer->setInterval(1000*(int)DEFAULT_CLEAR_RATE);

  m_GridLayoutClientControls = new QGridLayout(m_Frame);
  m_GridLayoutClientControls->setSpacing(0);
  m_GridLayoutClientControls->setContentsMargins(0, 0, 0, 0);

  m_Frame->setContentsMargins(0, 0, 0, 0);

  m_DirectoryChooser->setFilters(ctkPathLineEdit::Dirs);
  m_DirectoryChooser->setOptions(ctkPathLineEdit::ShowDirsOnly);

  // BEWARE: these need to be ordered in this way! various other code simply checks for
  //         selection index to distinguish between source types, etc.
  m_SourceSelectComboBox->addItem("networked tracker", mitk::IGIDataSource::SOURCE_TYPE_TRACKER);
  m_SourceSelectComboBox->addItem("networked ultrasonix scanner", mitk::IGIDataSource::SOURCE_TYPE_IMAGER);
  m_SourceSelectComboBox->addItem("local frame grabber", mitk::IGIDataSource::SOURCE_TYPE_FRAME_GRABBER);

#ifdef _USE_NVAPI
  m_SourceSelectComboBox->addItem("local NVidia SDI", mitk::IGIDataSource::SOURCE_TYPE_NVIDIA_SDI);
#endif

  m_ToolManagerPlaybackGroupBox->setCollapsed(true);
  m_ToolManagerConsoleGroupBox->setCollapsed(true);
  m_ToolManagerConsole->setMaximumHeight(100);
  m_TableWidget->setMaximumHeight(150);
  m_TableWidget->setSelectionMode(QAbstractItemView::SingleSelection);

  connect(m_SourceSelectComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnCurrentIndexChanged(int)));
  connect(m_AddSourcePushButton, SIGNAL(clicked()), this, SLOT(OnAddSource()) );
  connect(m_RemoveSourcePushButton, SIGNAL(clicked()), this, SLOT(OnRemoveSource()) );
  connect(m_TableWidget, SIGNAL(cellDoubleClicked(int, int)), this, SLOT(OnCellDoubleClicked(int, int)) );
  connect(m_RecordPushButton, SIGNAL(clicked()), this, SLOT(OnRecordStart()) );
  connect(m_StopPushButton, SIGNAL(clicked()), this, SLOT(OnRecordStop()) );
  connect(m_PlayPushButton, SIGNAL(clicked()), this, SLOT(OnPlayStart()));
  connect(m_GuiUpdateTimer, SIGNAL(timeout()), this, SLOT(OnUpdateGui()));
  connect(m_ClearDownTimer, SIGNAL(timeout()), this, SLOT(OnCleanData()));
  // FIXME: do we need to connect a slot to the playback slider? gui-update-timer will eventually check its state anyway.

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
int QmitkIGIDataSourceManager::GetSourceNumberFromIdentifier(int identifier)
{
  int sourceNumber = -1;
  for (unsigned int i = 0; i < m_Sources.size(); i++)
  {
    if (m_Sources[i].IsNotNull() && m_Sources[i]->GetIdentifier() == identifier)
    {
      sourceNumber = i;
      break;
    }
  }
  return sourceNumber;
}


//-----------------------------------------------------------------------------
int QmitkIGIDataSourceManager::GetIdentifierFromSourceNumber(int sourceNumber)
{
  int identifier = -1;
  if (sourceNumber < (int)m_Sources.size())
  {
    identifier = m_Sources[sourceNumber]->GetIdentifier();
  }
  return identifier;
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnAddSource()
{
  mitk::IGIDataSource::SourceTypeEnum sourceType =
      static_cast<mitk::IGIDataSource::SourceTypeEnum>(
          m_SourceSelectComboBox->itemData(m_SourceSelectComboBox->currentIndex()).toInt()
          );
  int portNumber = m_PortNumberSpinBox->value();

  if (this->IsPortSpecificType())
  {
    if (m_PortsInUse.contains(portNumber))
    {
      QMessageBox msgBox(QMessageBox::Warning, tr("Server failure"), tr("Could not open socket: already listening on the selected port!"), QMessageBox::Ok);
      msgBox.exec();
      return;
    }
  }
  int identifier = this->AddSource(sourceType, portNumber);

  // Force an update.
  this->UpdateSourceView(identifier, false);

  // Launch timers
  if (!m_GuiUpdateTimer->isActive())
  {
    m_GuiUpdateTimer->start();
  }
  if (!m_ClearDownTimer->isActive())
  {
    m_ClearDownTimer->start();
  }
}


//------------------------------------------------
int QmitkIGIDataSourceManager::AddSource(const mitk::IGIDataSource::SourceTypeEnum& sourceType, int portNumber, NiftyLinkSocketObject* socket)
{
  QmitkIGIDataSource::Pointer source = NULL;

  if (sourceType == mitk::IGIDataSource::SOURCE_TYPE_TRACKER || sourceType == mitk::IGIDataSource::SOURCE_TYPE_IMAGER)
  {
    QmitkIGINiftyLinkDataSource::Pointer niftyLinkSource = NULL;
    if (sourceType == mitk::IGIDataSource::SOURCE_TYPE_TRACKER)
    {
      niftyLinkSource = QmitkIGITrackerSource::New(m_DataStorage, socket);
    }
    else if (sourceType == mitk::IGIDataSource::SOURCE_TYPE_IMAGER)
    {
      niftyLinkSource = QmitkIGIUltrasonixTool::New(m_DataStorage, socket);
    }
    
    if (niftyLinkSource->ListenOnPort(portNumber))
    {
      m_PortsInUse.insert(portNumber);
    }
    source = niftyLinkSource;
  }
  else if (sourceType == mitk::IGIDataSource::SOURCE_TYPE_FRAME_GRABBER)
  {
    source = QmitkIGIOpenCVDataSource::New(m_DataStorage);
  }
#ifdef _USE_NVAPI
  else if (sourceType == mitk::IGIDataSource::SOURCE_TYPE_NVIDIA_SDI)
  {
    source = QmitkIGINVidiaDataSource::New(m_DataStorage);
  }
#endif
  else
  {
    std::cerr << "Matt, not implemented yet" << std::endl;
  }

  source->SetSourceType(sourceType);
  source->SetIdentifier(m_NextSourceIdentifier);
  source->SetTimeStampTolerance(m_TimingTolerance);

  m_NextSourceIdentifier++;
  m_Sources.push_back(source);

  connect(source.GetPointer(), SIGNAL(DataSourceStatusUpdated(int)), this, SLOT(OnUpdateSourceView(int)), Qt::QueuedConnection);

  return source->GetIdentifier();
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnRemoveSource()
{
  if (m_TableWidget->rowCount() == 0)
    return;

  // Stop the timers to make sure they don't trigger.
  bool guiTimerWasOn = m_GuiUpdateTimer->isActive();
  bool clearDownTimerWasOn = m_ClearDownTimer->isActive();
  m_GuiUpdateTimer->stop();
  m_ClearDownTimer->stop();

  // Get a valid row number, or delete the last item in the table.
  int rowIndex = m_TableWidget->currentRow();
  if (rowIndex < 0)
  {
    rowIndex = m_TableWidget->rowCount() - 1;
  }

  // scoping for source smart pointer
  {
    std::vector<int> rowsToDelete;
    rowsToDelete.push_back(rowIndex);

    QmitkIGIDataSource::Pointer source = m_Sources[rowIndex];

    // FIXME: The list of RelatedSources is kept in mitkIGIDataSource, so the idea of having
    // linked sources is not unique to network tools. So we either move the "related sources"
    // down the class hierarchy, so that it only applies to networked tools, or we generalise
    // this to destroy any related data source.

    // If it is a networked tool, scan for all sources using the same port.
    QmitkIGINiftyLinkDataSource::Pointer niftyLinkSource = dynamic_cast<QmitkIGINiftyLinkDataSource*>(source.GetPointer());
    if (niftyLinkSource.IsNotNull())
    {
      int portNumber = niftyLinkSource->GetPort();
      for ( int i = 0 ; i < m_TableWidget->rowCount() ; i ++ )
      {
        if (i != rowIndex)
        {
          QmitkIGIDataSource::Pointer tempSource = m_Sources[i];
          QmitkIGINiftyLinkDataSource::Pointer tempNiftyLinkSource = dynamic_cast<QmitkIGINiftyLinkDataSource*>(tempSource.GetPointer());
          if ( tempNiftyLinkSource.IsNotNull() )
          {
            int tempPortNumber = tempNiftyLinkSource->GetPort();
            if ( tempPortNumber == portNumber )
            {
              rowsToDelete.push_back(i);
            }
          }
        }
      } // end for
      m_PortsInUse.remove(portNumber);
    }

    std::sort(rowsToDelete.begin(), rowsToDelete.end(), std::less<int>());

    // Now delete all these sources.
    int numberDeleted = 0;
    for (unsigned int i = 0; i < rowsToDelete.size(); i++)
    {
      int actualRowNumber = rowsToDelete[i] - numberDeleted;
      source = m_Sources[actualRowNumber];
      disconnect(source, 0, this, 0);

      // Only remove current GUI, if the current GUI is connected to
      // a source that we are trying to delete.
      if (m_GridLayoutClientControls != NULL)
      {
        QLayoutItem *item = m_GridLayoutClientControls->itemAtPosition(0,0);
        if (item != NULL)
        {
          QWidget *widget = item->widget();
          if (widget != NULL)
          {
            QmitkIGIDataSourceGui *gui = dynamic_cast<QmitkIGIDataSourceGui*>(widget);
            if (gui != NULL && gui->GetSource() == source)
            {
              this->DeleteCurrentGuiWidget();
            }
          }
        }
      }

      m_Sources.erase(m_Sources.begin() + actualRowNumber);

      m_TableWidget->removeRow(actualRowNumber);
      m_TableWidget->update();

      numberDeleted++;
    }
  }

  // Given we stopped the timers to make sure they don't trigger, we need
  // to restart them, if indeed they were on.
  if (m_TableWidget->rowCount() > 0)
  {
    if (guiTimerWasOn)
    {
      m_GuiUpdateTimer->start();
    }
    if (clearDownTimerWasOn)
    {
      m_ClearDownTimer->start();
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnCellDoubleClicked(int row, int column)
{
  QmitkIGIDataSourceGui* sourceGui = NULL;

  mitk::IGIDataSource* source = m_Sources[row];
  const std::string classname = source->GetNameOfClass();

  std::string guiClassname = classname + "Gui";

  this->PrintStatusMessage(QString("INFO:Creating ") + QString::fromStdString(guiClassname) + QString(" for ") + QString::fromStdString(classname));

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
    this->DeleteCurrentGuiWidget();

    m_GridLayoutClientControls = new QGridLayout(m_Frame);
    m_GridLayoutClientControls->setSpacing(0);
    m_GridLayoutClientControls->setContentsMargins(0, 0, 0, 0);

    sourceGui->SetDataSource(source);
    sourceGui->Initialize(NULL);

    m_GridLayoutClientControls->addWidget(sourceGui, 0, 0);
  }
  else
  {
    this->PrintStatusMessage(QString("ERROR:For class ") + QString::fromStdString(classname) + ", could not create GUI class " + QString::fromStdString(guiClassname));
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::InstantiateRelatedSources(const int& rowNumber)
{
  // This method should only be called from UpdateSourceView, so we are assuming rowNumber is a valid array index.

  QmitkIGIDataSource::Pointer         source = m_Sources[rowNumber];
  mitk::IGIDataSource::SourceTypeEnum sourceType = m_Sources[rowNumber]->GetSourceType();
  std::string                         status = m_Sources[rowNumber]->GetStatus();
  std::string                         displayType = m_Sources[rowNumber]->GetType();
  std::string                         description = m_Sources[rowNumber]->GetDescription();
  std::string                         device = m_Sources[rowNumber]->GetName();
  std::list<std::string>              subSources = m_Sources[rowNumber]->GetRelatedSources();

  foreach ( std::string subSource, subSources )
  {
    // The related sources contains the list of all sub-sources from a given source.
    if (subSource != description)
    {

      bool createdAlready = false;
      for (unsigned int i = 0; i < m_Sources.size(); i++)
      {
        if (m_Sources[i]->GetSourceType() == sourceType
            && m_Sources[i]->GetType() == displayType
            && m_Sources[i]->GetDescription() == subSource
            && m_Sources[i]->GetName() == device
            )
        {
          createdAlready = true;
        }
      }
      if ( !createdAlready
           && (sourceType == mitk::IGIDataSource::SOURCE_TYPE_TRACKER
               || sourceType == mitk::IGIDataSource::SOURCE_TYPE_IMAGER
               )
         )
      {
        QmitkIGINiftyLinkDataSource::Pointer niftyLinkSource = dynamic_cast< QmitkIGINiftyLinkDataSource*>(source.GetPointer());

        int tempToolIdentifier = AddSource (sourceType, niftyLinkSource->GetPort(), niftyLinkSource->GetSocket());
        int tempRowNumber = this->GetSourceNumberFromIdentifier(tempToolIdentifier);

        m_Sources[tempRowNumber]->SetType(displayType);
        m_Sources[tempRowNumber]->SetName(device);
        m_Sources[tempRowNumber]->SetDescription(subSource);
        m_Sources[tempRowNumber]->SetStatus(status);
        this->UpdateSourceView(tempToolIdentifier, false);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::UpdateSourceView(const int& sourceIdentifier, bool instantiateRelatedSources)
{
  // Assumption:
  // rowNumber == sourceIdentifier, i.e. same thing, and should be a valid array index into m_Sources.

  int rowNumber = this->GetSourceNumberFromIdentifier(sourceIdentifier);
  assert(rowNumber >= 0);
  assert(rowNumber < (int)m_Sources.size());

  bool        update = m_Sources[rowNumber]->GetShouldCallUpdate();
  std::string status = m_Sources[rowNumber]->GetStatus();
  std::string type = m_Sources[rowNumber]->GetType();
  std::string device = m_Sources[rowNumber]->GetName();
  std::string description = m_Sources[rowNumber]->GetDescription();

  std::vector<std::string> fields;
  fields.push_back(status);
  fields.push_back(type);
  fields.push_back(device);
  fields.push_back(description);

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
  m_TableWidget->item(rowNumber, 0)->setFlags(m_TableWidget->item(rowNumber, 0)->flags() | Qt::ItemIsUserCheckable);
  m_TableWidget->item(rowNumber, 0)->setCheckState(update ? Qt::Checked : Qt::Unchecked);

  if (instantiateRelatedSources)
  {
    this->InstantiateRelatedSources(rowNumber);
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnUpdateSourceView(const int& sourceIdentifier)
{
  this->UpdateSourceView(sourceIdentifier, true);
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnUpdateGui()
{

  // whether we are currently grabbing live data or playing back canned bits
  // depends solely on the state of the play-button.
  if (m_PlayPushButton->isChecked())
  {
    int         sliderValue = m_PlaybackSlider->value();
    igtlUint64  sliderTime  = m_PlaybackSliderBase + ((double) sliderValue / m_PlaybackSliderFactor);

    m_CurrentTime = sliderTime;
  }
  else
  {
    igtl::TimeStamp::Pointer timeNow = igtl::TimeStamp::New();
    m_CurrentTime = GetTimeInNanoSeconds(timeNow);
  }

  igtlUint64 idNow = m_CurrentTime;
  emit UpdateGuiStart(idNow);

  if (m_Sources.size() > 0)
  {
    // Iterate over all sources, so where we have linked sources,
    // such as a tracker, tracking multiple tools, we have one row for
    // each tool. So each tool is a separate source.
    foreach ( QmitkIGIDataSource::Pointer source, m_Sources )
    {
      // Work out the sourceNumber == rowNumber.
      int rowNumber = this->GetSourceNumberFromIdentifier(source->GetIdentifier());

      // side note: communicating this ShouldCallUpdate is a one-way thing, gui to datasource.
      // i.e. the data source cannot set ShouldCallUpdate itself and expect the rest to do the right thing.
      // gui is in charge of controlling this flag.
      bool  shouldUpdate = m_TableWidget->item(rowNumber, 0)->checkState() == Qt::Checked;
      source->SetShouldCallUpdate(shouldUpdate);

      // First tell each source to update data.
      // For example, sources could copy to data storage.
      bool isValid = false;
      float rate = 0;
      double lag = 0;

      isValid = source->ProcessData(idNow);
      rate = source->UpdateFrameRate();
      lag = source->GetCurrentTimeLag(idNow);

      // Update the frame rate number.
      QTableWidgetItem *frameRateItem = new QTableWidgetItem(QString::number(rate));
      frameRateItem->setTextAlignment(Qt::AlignCenter);
      frameRateItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
      m_TableWidget->setItem(rowNumber, 4, frameRateItem);

      // Update the lag number.
      QTableWidgetItem *lagItem = new QTableWidgetItem(QString::number(lag));
      lagItem->setTextAlignment(Qt::AlignCenter);
      lagItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
      m_TableWidget->setItem(rowNumber, 5, lagItem);

      // Update the status icon.
      QTableWidgetItem *tItem = m_TableWidget->item(rowNumber, 0);
      if (!shouldUpdate)
      {
        QPixmap pix(22, 22);
        pix.fill(m_SuspendedColour);
        tItem->setIcon(pix);
      }
      else
      {
        if (!isValid || lag > m_TimingTolerance/1000000000) // lag is in seconds, timing tolerance in nanoseconds.
        {
          // Highlight that current row is in error.
          QPixmap pix(22, 22);
          pix.fill(m_ErrorColour);
          tItem->setIcon(pix);
        }
        else
        {
          // Highlight that current row is OK.
          QPixmap pix(22, 22);
          pix.fill(m_OKColour);
          tItem->setIcon(pix);
        }
      }
      // Update the status text.
      tItem->setText(QString::fromStdString(source->GetStatus()));
      tItem->setCheckState(shouldUpdate ? Qt::Checked : Qt::Unchecked);
    }

    emit UpdateGuiFinishedDataSources(idNow);

    // Make sure table is refreshing.
    m_TableWidget->update();

    // Make sure scene rendered.
    mitk::RenderingManager * renderer = mitk::RenderingManager::GetInstance();
    renderer->RequestUpdateAll();

    emit UpdateGuiFinishedFinishedRendering(idNow);

    // Try to encourage rest of event loop to process before the timer swamps it.
    //QCoreApplication::processEvents();
  }
  emit UpdateGuiEnd(idNow);
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnCleanData()
{
  // If gui timer is not running, we probably have no sources active.
  if (!m_GuiUpdateTimer->isActive())
  {
    return;
  }

  // If we are active, then simply ask each buffer to clean up in turn.
  foreach ( QmitkIGIDataSource::Pointer source, m_Sources )
  {
    source->CleanBuffer();
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnRecordStart()
{
  // Generate a unique directory folder name.
  QString baseDirectory = m_DirectoryPrefix;

  igtl::TimeStamp::Pointer timeStamp = igtl::TimeStamp::New();

  igtlUint32 seconds;
  igtlUint32 nanoseconds;
  igtlUint64 millis;

  timeStamp->GetTime(&seconds, &nanoseconds);
  millis = (igtlUint64)seconds*1000 + nanoseconds/1000000;

  QDateTime dateTime;
  dateTime.setMSecsSinceEpoch(millis);

  QString formattedTime = dateTime.toString("yyyy-MM-dd-hh-mm-ss-zzz");

  QDir directory(baseDirectory + QDir::separator() + formattedTime);
  m_DirectoryChooser->setCurrentPath(directory.absolutePath());

  foreach ( QmitkIGIDataSource::Pointer source, m_Sources )
  {
    source->StartRecording(directory.absolutePath().toStdString(), this->m_SaveInBackground, this->m_SaveOnReceipt);
  }

  m_RecordPushButton->setEnabled(false);
  m_StopPushButton->setEnabled(true);
  assert(!m_PlayPushButton->isChecked());
  m_PlayPushButton->setEnabled(false);
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnRecordStop()
{
  foreach ( QmitkIGIDataSource::Pointer source, m_Sources )
  {
    source->StopRecording();
  }

  m_RecordPushButton->setEnabled(true);
  m_StopPushButton->setEnabled(false);
  assert(!m_PlayPushButton->isChecked());
  m_PlayPushButton->setEnabled(true);
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::OnPlayStart()
{
  if (m_PlayPushButton->isChecked())
  {
    QString playbackpath = m_DirectoryChooser->currentPath();
    // FIXME: playback button should only be enabled if there's a path in m_DirectoryChooser.
    if (playbackpath.isEmpty())
    {
      m_PlayPushButton->setChecked(false);
    }
    else
    {
      std::set<QmitkIGIDataSource::Pointer> goodSources;

      igtlUint64    overallStartTime = std::numeric_limits<igtlUint64>::max();
      igtlUint64    overallEndTime   = std::numeric_limits<igtlUint64>::min();
      std::string   pathstring       = playbackpath.toStdString();

      foreach (QmitkIGIDataSource::Pointer source, m_Sources)
      {
        igtlUint64  startTime = -1;
        igtlUint64  endTime   = -1;
        bool cando = source->ProbeRecordedData(pathstring, &startTime, &endTime);
        if (cando)
        {
          overallStartTime = std::min(overallStartTime, startTime);
          overallEndTime   = std::max(overallEndTime, endTime);

          goodSources.insert(source);
        }
        else
        {
          // data source that cannot playback enters freeze-frame mode
          source->SetShouldCallUpdate(false);
        }
      }

      if (overallEndTime >= overallStartTime)
      {
        foreach (QmitkIGIDataSource::Pointer source, goodSources)
        {
          source->ClearBuffer();
          source->StartPlayback(pathstring, overallStartTime, overallEndTime);
        }


        m_PlaybackSliderBase = overallStartTime;
        m_PlaybackSliderFactor = (std::numeric_limits<int>::max() / 2) / (double) (overallEndTime - overallStartTime);
        // if the time range is very short then dont upscale for the slider
        m_PlaybackSliderFactor = std::min(m_PlaybackSliderFactor, 1.0);

        double  sliderMax = m_PlaybackSliderFactor * (overallEndTime - overallStartTime);
        assert(sliderMax < std::numeric_limits<int>::max());

        m_PlaybackSlider->setMinimum(0);
        m_PlaybackSlider->setMaximum(sliderMax);

        // pop open the controls
        m_ToolManagerPlaybackGroupBox->setCollapsed(false);
        // can stop playback with stop button (in addition to unchecking the playbutton)
        m_StopPushButton->setEnabled(true);
        // for now, cannot start recording directly from playback mode.
        // could be possible: leave this enabled and simply stop all playback when user clicks on record.
        m_RecordPushButton->setEnabled(false);
      }
      else
      {
        m_PlayPushButton->setChecked(false);
      }
    }
  }
  else
  {
    foreach (QmitkIGIDataSource::Pointer source, m_Sources)
    {
      source->StopPlayback();
    }

    m_StopPushButton->setEnabled(false);
    m_RecordPushButton->setEnabled(true);
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceManager::PrintStatusMessage(const QString& message) const
{
  m_ToolManagerConsole->appendPlainText(message + "\n");
  MITK_INFO << "QmitkIGIDataSourceManager:" << message.toStdString() << std::endl;
}
