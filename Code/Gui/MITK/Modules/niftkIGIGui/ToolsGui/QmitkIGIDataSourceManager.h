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

#ifndef QMITKIGIDATASOURCEMANAGER_H
#define QMITKIGIDATASOURCEMANAGER_H

#include "niftkIGIGuiExports.h"
#include "ui_QmitkIGIDataSourceManager.h"
#include <itkObject.h>
#include <QWidget>
#include <QList>
#include <QGridLayout>
#include <QTimer>
#include <QSet>
#include <QColor>
#include <mitkDataStorage.h>
#include "mitkIGIDataSource.h"

class QmitkStdMultiWidget;

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

  mitkClassMacro(QmitkIGIDataSourceManager, itk::Object);
  itkNewMacro(QmitkIGIDataSourceManager);

  static QString GetDefaultPath();
  static const QColor DEFAULT_ERROR_COLOUR;
  static const QColor DEFAULT_WARNING_COLOUR;
  static const QColor DEFAULT_OK_COLOUR;
  static const int    DEFAULT_FRAME_RATE;

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

protected:

  QmitkIGIDataSourceManager();
  virtual ~QmitkIGIDataSourceManager();

  QmitkIGIDataSourceManager(const QmitkIGIDataSourceManager&); // Purposefully not implemented.
  QmitkIGIDataSourceManager& operator=(const QmitkIGIDataSourceManager&); // Purposefully not implemented.

private slots:

  /**
   * \brief Updates the display, based on the available messages.
   */
  void OnUpdateDisplay();

  /**
   * \brief Updates the frame rate.
   */
  void OnUpdateFrameRate();

  /**
   * \brief Adds a data source to the table.
   */
  void OnAddSource();

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
  QTimer                                   *m_UpdateTimer;
  QTimer                                   *m_FrameRateTimer;
  QSet<int>                                 m_PortsInUse;
  std::vector<mitk::IGIDataSource::Pointer> m_Sources;
  unsigned int                              m_NextSourceIdentifier;

  QColor                                    m_ErrorColour;
  QColor                                    m_WarningColour;
  QColor                                    m_OKColour;
  int                                       m_FrameRate;
  QString                                   m_DirectoryPrefix;

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

}; // end class

#endif

