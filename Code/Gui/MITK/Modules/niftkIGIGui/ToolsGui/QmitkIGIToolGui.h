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

#ifndef QMITKIGITOOLGUI_H
#define QMITKIGITOOLGUI_H

#include "niftkIGIGuiExports.h"
#include <itkObject.h>
#include <QWidget>
#include <mitkDataStorage.h>
#include "QmitkIGITool.h"

class ClientDescriptorXMLBuilder;
class QmitkStdMultiWidget;

/**
 * \class QmitkIGIToolGui
 * \brief Base class for the GUI belonging to an IGI Tool.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIToolGui : public QWidget, public itk::Object
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIToolGui, itk::Object);

  /**
   * \brief Sets the tool that this GUI will operate.
   */
  void SetTool(QmitkIGITool *tool);

  /**
   * \brief Retrieves the tool that this GUI will operate on.
   */
  itkGetConstMacro(Tool, QmitkIGITool*);

  /**
   * \brief Sets the StdMultiWidget.
   */
  itkSetObjectMacro(StdMultiWidget, QmitkStdMultiWidget);

  /**
   * \brief Gets the StdMultiWidget.
   */
  itkGetConstMacro(StdMultiWidget, QmitkStdMultiWidget*);

  /**
   * \brief ToolGui can have an optional Initialize function to perform any setup.
   */
  virtual void Initialize(QWidget *parent, ClientDescriptorXMLBuilder *config) {};

  // just make sure ITK won't take care of anything (especially not destruction)
  virtual void Register() const;
  virtual void UnRegister() const;
  virtual void SetReferenceCount(int);

signals:

  void NewToolAssociated( QmitkIGITool* );

protected:

  QmitkIGIToolGui(); // Purposefully hidden.
  virtual ~QmitkIGIToolGui(); // Purposefully hidden.

  QmitkIGIToolGui(const QmitkIGIToolGui&); // Purposefully not implemented.
  QmitkIGIToolGui& operator=(const QmitkIGIToolGui&); // Purposefully not implemented.

private:

  QmitkIGITool        *m_Tool;
  QmitkStdMultiWidget *m_StdMultiWidget;

}; // end class

#endif
