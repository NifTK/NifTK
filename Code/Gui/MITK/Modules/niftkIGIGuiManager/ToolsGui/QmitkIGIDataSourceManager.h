/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKIGIDATASOURCEMANAGER_H
#define QMITKIGIDATASOURCEMANAGER_H

#include "niftkIGIGuiManagerExports.h"
#include "ui_QmitkIGIDataSourceManager.h"
#include <itkObject.h>
#include <QWidget>
#include <QList>
#include <QGridLayout>
#include <QTimer>
#include <QSet>
#include <QColor>
#include <QThread>
#include <mitkDataStorage.h>
#include "mitkIGIDataSource.h"
#include <OIGTLSocketObject.h>

class QmitkStdMultiWidget;
class QmitkIGIDataSourceManagerClearDownThread;
class QmitkIGIDataSourceManagerGuiUpdateThread;

/**
 * \class QmitkIGIDataSourceManager
 * \brief Class to manage a list of QmitkIGIDataSources (trackers, ultra-sound machines, video etc).
 *
 * The SurgicalGuidanceView creates this widget to manage its tools. This widget acts like
 * a widget factory, setting up sockets, creating the appropriate widget, and instantiating
 * the appropriate GUI, and loading it into the grid layout owned by this widget.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIDataSourceManager : public QWidget, public Ui_QmitkIGIDataSourceManager, public itk::Object
{

  Q_OBJECT

public:

  friend class QmitkIGIDataSourceManagerClearDownThread;
  friend class QmitkIGIDataSourceManagerGuiUpdateThread;

  mitkClassMacro(QmitkIGIDataSourceManager, itk::Object);
  itkNewMacro(QmitkIGIDataSourceManager);

  static QString GetDefaultPath();
  static const QColor DEFAULT_ERROR_COLOUR;
  static const QColor DEFAULT_WARNING_COLOUR;
  static const QColor DEFAULT_OK_COLOUR;
  static const int    DEFAULT_FRAME_RATE;
  static const int    DEFAULT_CLEAR_RATE;
  static const bool   DEFAULT_SAVE_ON_RECEIPT;
  static const bool   DEFAULT_SAVE_IN_BACKGROUND;

  /**
   * \brief Creates the base class widgets, and connects signals and slots.
   */
  void setupUi(QWidget* parent);

  /*
   * \brief Set the Data Storage, and also sets it into any registered tools.
   * \param dataStorage An MITK DataStorage, which is set onto any registered tools.
   */
  void SetDataStorage(mitk::DataStorage* dataStorage);

  /**
   * \brief Get the Data Storage that this tool manager is currently connected to.
   */
  itkGetConstMacro(DataStorage, mitk::DataStorage*);

  /**
   * \brief Sets the StdMultiWidget.
   */
  itkSetObjectMacro(StdMultiWidget, QmitkStdMultiWidget);

  /**
   * \brief Gets the StdMultiWidget.
   */
  itkGetConstMacro(StdMultiWidget, QmitkStdMultiWidget*);

  /**
   * \brief Called from the GUI when the surgical guidance plugin preferences are modified.
   */
  void SetFramesPerSecond(int framesPerSecond);

  /**
   * \brief Called from the GUI when the surgical guidance plugin preferences are modified.
   */
  void SetClearDataRate(int numberOfSeconds);

  /**
   * \brief Called from the GUI when the surgical guidance plugin preferences are modified.
   */
  void SetDirectoryPrefix(QString& directoryPrefix);

  /**
   * \brief Called from the GUI when the surgical guidance plugin preferences are modified.
   */
  void SetErrorColour(QColor &colour);

  /**
   * \brief Called from the GUI when the surgical guidance plugin preferences are modified.
   */
  void SetWarningColour(QColor &colour);

  /**
   * \brief Called from the GUI when the surgical guidance plugin preferences are modified.
   */
  void SetOKColour(QColor &colour);

  /**
   * \brief Called from the GUI when the surgical guidance plugin preferences are modified.
   */
  itkSetMacro(SaveOnReceipt, bool);

  /**
   * \brief Called from the GUI when the surgical guidance plugin preferences are modified.
   */
  itkSetMacro(SaveInBackground, bool);

protected:

  QmitkIGIDataSourceManager();
  virtual ~QmitkIGIDataSourceManager();

  QmitkIGIDataSourceManager(const QmitkIGIDataSourceManager&); // Purposefully not implemented.
  QmitkIGIDataSourceManager& operator=(const QmitkIGIDataSourceManager&); // Purposefully not implemented.

private slots:

  /**
   * \brief Updates the whole rendered scene, based on the available messages.
   */
  void OnUpdateDisplay();

  /**
   * \brief Updates the frame rate.
   */
  void OnUpdateFrameRate();

  /**
   * \brief Tells each data source to clean data, see mitk::IGIDataSource::CleanData().
   */
  void OnCleanData();

  /**
   * \brief Adds a data source to the table, using values from UI for sourcetype and portnumbeR
   */
  void OnAddSource();

  /**
   * \brief Adds a data source to the table.
   */
  int AddSource(int sourcetype , int portnumber);

  /**
   * \brief Adds a data source to the table.
   * \return the added tool's identifier
   */
  int AddSource(int sourcetype , int portnumber, OIGTLSocketObject* socket);

  /**
   * \brief Removes a data source from the table, and completely destroys it.
   */
  void OnRemoveSource();

  /**
   * \brief Callback to indicate that a cell has been double clicked, to launch that sources' GUI.
   */
  void OnCellDoubleClicked(int row, int column);

  /**
   * \brief Callback when combo box for data source type is changed, we enable/disable widgets accordingly.
   */
  void OnCurrentIndexChanged(int indexNumber);

  /**
   * \brief Callback to start recording data.
   */
  void OnRecordStart();

  /**
   * \brief Callback to stop recording data.
   */
  void OnRecordStop();

private:

  mitk::DataStorage                        *m_DataStorage;
  QmitkStdMultiWidget                      *m_StdMultiWidget;
  QGridLayout                              *m_GridLayoutClientControls;
  QTimer                                   *m_FrameRateTimer; // Used to just update the frame rate
  QSet<int>                                 m_PortsInUse;
  std::vector<mitk::IGIDataSource::Pointer> m_Sources;
  unsigned int                              m_NextSourceIdentifier;

  QColor                                    m_ErrorColour;
  QColor                                    m_WarningColour;
  QColor                                    m_OKColour;
  int                                       m_FrameRate;
  int                                       m_ClearDataRate;
  QString                                   m_DirectoryPrefix;
  bool                                      m_SaveOnReceipt;
  bool                                      m_SaveInBackground;
  QmitkIGIDataSourceManagerClearDownThread *m_ClearDownThread;
  QmitkIGIDataSourceManagerGuiUpdateThread *m_GuiUpdateThread;

  /**
   * \brief Checks the m_SourceSelectComboBox to see if the currentIndex pertains to a port specific type.
   */
  bool IsPortSpecificType();

  /**
   * m_Sources is ordered, and MUST correspond to the order in the display QTableWidget,
   * so this returns the row number or -1 if not found.
   */
  int GetRowNumberFromIdentifier(int toolIdentifier);

  /**
   * m_Sources is ordered, and MUST correspond to the order in the display QTableWidget,
   * so this returns the identifier or -1 if not found.
   */
  int GetIdentifierFromRowNumber(int rowNumber);

  /**
   * \brief Works out the table row, then updates the fields in the GUI.
   */
  void UpdateToolDisplay(int toolIdentifier);

  /**
   * \brief Adds a message to the QmitkIGIDataSourceManager console.
   */
  void PrintStatusMessage(const QString& message) const;

  /**
   * \brief Deletes the current GUI widget.
   */
  void DeleteCurrentGuiWidget();

}; // end class

/**
 * \brief Separate thread class to run the clear down.
 */
class QmitkIGIDataSourceManagerClearDownThread : public QThread {
  Q_OBJECT
public:
  QmitkIGIDataSourceManagerClearDownThread(QObject *parent, QmitkIGIDataSourceManager *manager);
  ~QmitkIGIDataSourceManagerClearDownThread();

  void SetInterval(unsigned int milliseconds);
  void run();

public slots:
  void OnTimeout();

private:
  unsigned int m_TimerInterval;
  QTimer *m_Timer;
  QmitkIGIDataSourceManager *m_Manager;
};

/**
 * \brief Separate thread class to run the GUI update at the right rate.
 */
class QmitkIGIDataSourceManagerGuiUpdateThread : public QThread {
  Q_OBJECT
public:
  QmitkIGIDataSourceManagerGuiUpdateThread(QObject *parent, QmitkIGIDataSourceManager *manager);
  ~QmitkIGIDataSourceManagerGuiUpdateThread();

  void SetInterval(unsigned int milliseconds);
  void run();

public slots:
  void OnTimeout();

private:
  unsigned int m_TimerInterval;
  QTimer *m_Timer;
  QmitkIGIDataSourceManager *m_Manager;
};

#endif

