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

#ifndef QMITKIGIDATASOURCEGUI_H
#define QMITKIGIDATASOURCEGUI_H

#include "niftkIGIGuiExports.h"
#include <itkObject.h>
#include <QWidget>
#include <mitkDataStorage.h>
#include "mitkIGIDataSource.h"

class QmitkStdMultiWidget;

/**
 * \class QmitkIGIDataSourceGui
 * \brief Base class for the GUI belonging to an IGI Tool.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIDataSourceGui : public QWidget, public itk::Object
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIDataSourceGui, itk::Object);

  /**
   * \brief Sets the DataSource that this GUI will operate.
   */
  void SetDataSource(mitk::IGIDataSource *source);

  /**
   * \brief Retrieves the source that this GUI will operate on.
   */
  itkGetConstMacro(Source, mitk::IGIDataSource*);

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
  virtual void Initialize(QWidget *parent) {};

  // just make sure ITK won't take care of anything (especially not destruction)
  virtual void Register() const;
  virtual void UnRegister() const;
  virtual void SetReferenceCount(int);

signals:

  void NewSourceAssociated( mitk::IGIDataSource* );

protected:

  QmitkIGIDataSourceGui(); // Purposefully hidden.
  virtual ~QmitkIGIDataSourceGui(); // Purposefully hidden.

  QmitkIGIDataSourceGui(const QmitkIGIDataSourceGui&); // Purposefully not implemented.
  QmitkIGIDataSourceGui& operator=(const QmitkIGIDataSourceGui&); // Purposefully not implemented.

private:

  mitk::IGIDataSource *m_Source;
  QmitkStdMultiWidget *m_StdMultiWidget;

}; // end class

#endif
