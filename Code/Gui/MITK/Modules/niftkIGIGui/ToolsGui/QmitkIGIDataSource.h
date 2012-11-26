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

#ifndef QMITKIGIDATASOURCE_H
#define QMITKIGIDATASOURCE_H

#include "niftkIGIGuiExports.h"
#include <QObject>
#include <QPixmap>
#include <itkObject.h>
#include <itkVersion.h>
#include <itkObjectFactoryBase.h>
#include <mitkDataStorage.h>

/**
 * \class QmitkIGIDataSource
 * \brief Base class for IGI Data Sources, where Data Sources can be video data,
 * ultrasound data, tracker data etc.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIDataSource : public QObject, public itk::Object
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIDataSource, itk::Object);

  /**
   * \brief Sets the identifier, which is just a tag to identify the tool by (i.e. item in a list).
   */
  itkSetMacro(Identifier, int);
  itkGetMacro(Identifier, int);

  /**
   * \brief Sets a name, useful for display purposes.
   */
  itkSetMacro(Name, std::string);
  itkGetMacro(Name, std::string);

  /**
   * \brief Sets a type, useful for display purposes.
   */
  itkSetMacro(Type, std::string);
  itkGetMacro(Type, std::string);

  /**
   * \brief Sets a description, useful for display purposes.
   */
  itkSetMacro(Description, std::string);
  itkGetMacro(Description, std::string);

  /**
   * \brief Derived classes can return a colour coded Icon of 22x22 pixels to describe the current status.
   */
  QPixmap GetStatusIcon() const;

  /**
   * \brief Framerate is calculated internally, and can be retrieved here.
   */
  itkGetMacro(FrameRate, float);

  /**
   * \brief Sets the data storage.
   */
  itkSetObjectMacro(DataStorage, mitk::DataStorage);

  /**
   * \brief Retrieves the data storage.
   */
  itkGetConstMacro(DataStorage, mitk::DataStorage*);

  /**
   * \brief Tools can have an optional Initialize function to perform any setup after construction,
   * with this class providing a default, do-nothing implementation.
   */
  virtual void Initialize() {};

public slots:

signals:

  /**
   * \brief Each tool should signal when the status has updated, so GUI can redraw, passing its internal identifier.
   */
  void DataSourceStatusUpdated(int identifier);

protected:

  QmitkIGIDataSource(); // Purposefully hidden.
  virtual ~QmitkIGIDataSource(); // Purposefully hidden.

  QmitkIGIDataSource(const QmitkIGIDataSource&); // Purposefully not implemented.
  QmitkIGIDataSource& operator=(const QmitkIGIDataSource&); // Purposefully not implemented.

  /**
   * \brief Derived classes can set the frame rate.
   */
  itkSetMacro(FrameRate, float);

  /**
   * \brief Derived classes should update the frame rate, as they receive data.
   */
  virtual void UpdateFrameRate() { m_FrameRate = 0; }

private:

  mitk::DataStorage *m_DataStorage;
  int                m_Identifier;
  float              m_FrameRate;
  std::string        m_Name;
  std::string        m_Type;
  std::string        m_Description;

}; // end class

#endif

