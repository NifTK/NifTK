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

#ifndef QMITKIGITOOL_H
#define QMITKIGITOOL_H

#include "niftkIGIGuiExports.h"
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <mitkDataStorage.h>
#include <QObject>
#include <OIGTLSocketObject.h>
#include <Common/NiftyLinkXMLBuilder.h>
#include "QmitkIGIToolFactoryMacro.h"

/**
 * \class QmitkIGITool
 * \brief Base class for IGI Tools.
 *
 * This class does not own the data storage, socket, or client descriptor,
 * so do not try and delete them. Furthermore, a tool must not know anything
 * about the GUI connected to it. It should function independently of
 * whether it is on-screen or not. For this reason we derive from QObject.
 * The GUI however can have a pointer to the tool.
 *
 * \see QmitkIGIToolGui
 */
class NIFTKIGIGUI_EXPORT QmitkIGITool : public QObject, public itk::Object
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGITool, itk::Object);

  /**
   * \brief Sets the data storage onto the tool.
   */
  itkSetObjectMacro(DataStorage, mitk::DataStorage);

  /**
   * \brief Retrieves the data storage that the tool is currently connected to.
   */
  itkGetConstMacro(DataStorage, mitk::DataStorage*);

  /**
   * \brief Sets the socket pointer.
   */
  itkSetObjectMacro(Socket, OIGTLSocketObject);

  /**
   * \brief Gets the socket pointer.
   */
  itkGetConstMacro(Socket, OIGTLSocketObject*);

  /**
   * \brief Sets the Client Descriptor XML.
   */
  itkSetObjectMacro(ClientDescriptor, ClientDescriptorXMLBuilder);

  /**
   * \brief Gets the Client Descriptor XML.
   */
  itkGetConstMacro(ClientDescriptor, ClientDescriptorXMLBuilder*);

  /**
   * \brief Returns the port number that this tool is using or -1 if no socket is available.
   */
  int GetPort() const;

  /**
   * \brief If there is a socket associated with this tool, will send the message.
   */
  void SendMessage(OIGTLMessage::Pointer msg);

  /**
   * \brief Tools can have an optional Initialize function to perform any setup after construction,
   * with this class providing a default, do-nothing implementation.
   */
  virtual void Initialize() {};

public slots:

  /**
   * \brief Main message handler routine for this tool, that subclasses must implement.
   */
  virtual void InterpretMessage(OIGTLMessage::Pointer msg) = 0;

protected:

  QmitkIGITool(); // Purposefully hidden.
  virtual ~QmitkIGITool(); // Purposefully hidden.

  QmitkIGITool(const QmitkIGITool&); // Purposefully not implemented.
  QmitkIGITool& operator=(const QmitkIGITool&); // Purposefully not implemented.

private:

  mitk::DataStorage           *m_DataStorage;
  OIGTLSocketObject           *m_Socket;
  ClientDescriptorXMLBuilder  *m_ClientDescriptor;

}; // end class

#endif

