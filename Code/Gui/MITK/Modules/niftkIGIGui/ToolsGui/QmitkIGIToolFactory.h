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

#ifndef QMITKIGITOOLFACTORY_H
#define QMITKIGITOOLFACTORY_H

#include "niftkIGIGuiExports.h"
#include <itkObject.h>
#include <QObject>
#include <mitkDataStorage.h>
#include "QmitkIGITool.h"
#include "QmitkIGIToolGui.h"

/**
 * \class QmitkIGIToolFactory
 * \brief Contains logic for instantiating QmitkIGITools and QmitkIGIToolGuis.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIToolFactory : public QObject, public itk::Object
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIToolFactory, itk::Object);
  itkNewMacro(QmitkIGIToolFactory);

  /**
   * \brief Instantiates a tool based on the client descriptor.
   */
  virtual QmitkIGITool::Pointer CreateTool(ClientDescriptorXMLBuilder& descriptor);

  /**
   * \brief Instantiates a tool GUI based on the tool, and the specified prefix and postfix.
   */
  virtual QmitkIGIToolGui::Pointer CreateGUI(QmitkIGITool* tool, const QString& prefix, const QString& postfix);

protected:

  QmitkIGIToolFactory(); // Purposefully hidden.
  virtual ~QmitkIGIToolFactory(); // Purposefully hidden.

  QmitkIGIToolFactory(const QmitkIGIToolFactory&); // Purposefully not implemented.
  QmitkIGIToolFactory& operator=(const QmitkIGIToolFactory&); // Purposefully not implemented.

};
#endif
