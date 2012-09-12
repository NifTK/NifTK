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

#ifndef QMITKIGITOOLMANAGER_H
#define QMITKIGITOOLMANAGER_H

#include "niftkIGIGuiExports.h"
#include "ui_QmitkIGIToolManager.h"
#include <itkObject.h>
#include <QWidget>
#include <QList>
#include <QGridLayout>
#include <mitkDataStorage.h>
#include "QmitkIGIToolFactory.h"

class QmitkIGITool;
class OIGTLSocketObject;
class XMLBuilderBase;
class QmitkStdMultiWidget;

/**
 * \class QmitkIGIToolManager
 * \brief Class to manage a list of QmitkIGITools (trackers, ultra-sound machines, video etc).
 *
 * The SurgicalGuidanceView creates this widget to manage its tools. This widget acts like
 * a widget factory, setting up sockets, creating the appropriate widget, and instantiating
 * the appropriate GUI, and loading it into the grid layout owned by this widget.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIToolManager : public QWidget, public Ui_QmitkIGIToolManager, public itk::Object
{

  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIToolManager, itk::Object);
  itkNewMacro(QmitkIGIToolManager);

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

protected:

  QmitkIGIToolManager();
  virtual ~QmitkIGIToolManager();

  QmitkIGIToolManager(const QmitkIGIToolManager&); // Purposefully not implemented.
  QmitkIGIToolManager& operator=(const QmitkIGIToolManager&); // Purposefully not implemented.

private slots:

  /**
   * \brief Called when the user clicks the "+" button, to add a new IGI tool.
   */
  void OnAddListeningPort();

  /**
   * \brief Called when the user clicks the "-" button, to deregister and destroy an IGI tool.
   */
  void OnRemoveListeningPort();

  /**
   * \brief If the user selects a different row, the spin box is updated with the right port number.
   */
  void OnTableSelectionChange(int r, int c, int pr = 0, int pc = 0);

  /**
   * \brief When the user clicks on a row, the appropriate GUI is displayed.
   */
  void OnCellDoubleClicked(int r, int c);

  /**
   * \brief This slot is triggered when a client connects to the local server, it changes the UI accordingly.
   */
  void ClientConnected();

  /**
   * \brief This slot is triggered when a client disconnects from the local server, it changes the UI accordingly.
   */
  void ClientDisconnected();

  /**
   * \brief This function interprets the received OIGTL messages, but only processes the initial client connected message.
   */
  void InterpretMessage(OIGTLMessage::Pointer msg);

private:

  mitk::DataStorage                       *m_DataStorage;
  QmitkStdMultiWidget                     *m_StdMultiWidget;
  QGridLayout                             *m_GridLayoutClientControls;
  QmitkIGIToolFactory::Pointer             m_ToolFactory;
  QHash<int, OIGTLSocketObject*>           m_Sockets;
  QHash<int, ClientDescriptorXMLBuilder *> m_ClientDescriptors;
  QHash<int, QmitkIGITool::Pointer>        m_Tools;
}; // end class

#endif

