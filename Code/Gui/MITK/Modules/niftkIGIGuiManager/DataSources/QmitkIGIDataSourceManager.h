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
#include <mitkIGIDataSource.h>
#include <NiftyLinkSocketObject.h>
#include "QmitkIGIDataSource.h"

class QmitkStdMultiWidget;
class QmitkIGIDataSourceManagerClearDownThread;
class QTimer;
class QGridLayout;

/**
 * \class QmitkIGIDataSourceManager
 * \brief Class to manage a list of QmitkIGIDataSources (trackers, ultra-sound machines, video etc).
 *
 * This widget acts like a widget factory, setting up sources, instantiating
 * the appropriate GUI, and loading it into the grid layout owned by this widget.
 */
class NIFTKIGIGUIMANAGER_EXPORT QmitkIGIDataSourceManager : public QWidget, public Ui_QmitkIGIDataSourceManager, public itk::Object
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

  /**
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

signals:

  /**
   * \brief Emmitted as soon as the OnUpdateGui method has acquired a timestamp.
   */
  void UpdateGuiStart(igtlUint64 timeStamp);

  /**
   * \brief Emmitted when the OnUpdateGui method has asked each data source to update.
   */
  void UpdateGuiFinishedDataSources(igtlUint64 timeStamp);

  /**
   * \brief Emmitted when the OnUpdateGui method has updated a few widgets, and called the rendering manager.
   */
  void UpdateGuiFinishedFinishedRendering(igtlUint64 timeStamp);

  /**
   * \brief Emmitted when the OnUpdateGui method has finished, after the QCoreApplication::processEvents() has been called.
   */
  void UpdateGuiEnd(igtlUint64 timeStamp);

protected:

  QmitkIGIDataSourceManager();
  virtual ~QmitkIGIDataSourceManager();

  QmitkIGIDataSourceManager(const QmitkIGIDataSourceManager&); // Purposefully not implemented.
  QmitkIGIDataSourceManager& operator=(const QmitkIGIDataSourceManager&); // Purposefully not implemented.

private slots:

  /**
   * \brief Updates the whole rendered scene, based on the available messages.
   *
   * More specifically, this method is called on a timer, and determines the
   * effective refresh rate of the data storage, and hence of the screen,
   * and also the widgets of the QmitkDataSourceManager itself. This method
   * assumes that all the data sources are instantiated, and the right number
   * of rows exists in the table.
   */
  void OnUpdateGui();

  /**
   * \brief Works out the table row, then updates the fields in the GUI.
   *
   * In comparison with OnUpdateGui, this method is just for updating the table
   * of available sources. Importantly, there is a use case where we need to dynamically
   * add rows. When a tool (typically a networked tool) provides information that
   * there should be additional related sources, we have to dynamically create them.
   *
   * \see OnUpdateGui
   */
  void OnUpdateSourceView(const int& sourceIdentifier);

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
   * \return the added tool's identifier
   */
  int AddSource(const mitk::IGIDataSource::SourceTypeEnum& sourcetype, int portnumber, NiftyLinkSocketObject* socket=NULL);

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
  QSet<int>                                 m_PortsInUse;
  std::vector<QmitkIGIDataSource::Pointer>  m_Sources;
  unsigned int                              m_NextSourceIdentifier;

  QColor                                    m_ErrorColour;
  QColor                                    m_WarningColour;
  QColor                                    m_OKColour;
  int                                       m_FrameRate;
  int                                       m_ClearDataRate;
  QString                                   m_DirectoryPrefix;
  bool                                      m_SaveOnReceipt;
  bool                                      m_SaveInBackground;
  QTimer                                   *m_GuiUpdateTimer;
  QTimer                                   *m_ClearDownTimer;

  /**
   * \brief Checks the m_SourceSelectComboBox to see if the currentIndex pertains to a port specific type.
   */
  bool IsPortSpecificType();

  /**
   * m_Sources is ordered, and MUST correspond to the order in the display QTableWidget,
   * so this returns the source number or -1 if not found.
   */
  int GetSourceNumberFromIdentifier(int sourceIdentifier);

  /**
   * m_Sources is ordered, and MUST correspond to the order in the display QTableWidget,
   * so this returns the identifier or -1 if not found.
   */
  int GetIdentifierFromSourceNumber(int sourceNumber);

  /**
   * \brief Called by UpdateSourceView to actually instantiate the extra rows needed dynamically.
   */
  void InstantiateRelatedSources(const int& rowNumber);

  /**
   * \brief Adds a message to the QmitkIGIDataSourceManager console.
   */
  void PrintStatusMessage(const QString& message) const;

  /**
   * \brief Deletes the current GUI widget.
   */
  void DeleteCurrentGuiWidget();

}; // end class

#endif

