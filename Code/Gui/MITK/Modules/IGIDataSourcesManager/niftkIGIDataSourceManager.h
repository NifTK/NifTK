/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceManager_h
#define niftkIGIDataSourceManager_h

#include "niftkIGIDataSourcesManagerExports.h"
#include "ui_niftkIGIDataSourceManager.h"

#include <mitkCommon.h>
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

#include <QWidget>

namespace niftk
{

/**
 * \class IGIDataSourceManager
 * \brief Class to manage a list of IGIDataSources (trackers, ultra-sound machines, video etc).
 *
 * This widget acts like a widget factory, setting up sources, instantiating
 * the appropriate GUI, and loading it into the grid layout owned by this widget.
 */
class NIFTKIGIDATASOURCESMANAGER_EXPORT IGIDataSourceManager : public QWidget, public Ui_IGIDataSourceManager, public itk::Object
{

  Q_OBJECT

public:

  mitkClassMacroItkParent(IGIDataSourceManager, itk::Object);
  itkNewMacro(IGIDataSourceManager);

  /**
   * \brief Creates the base class widgets, and connects signals and slots.
   */
  void setupUi(QWidget* parent);

protected:

  IGIDataSourceManager();
  virtual ~IGIDataSourceManager();

  IGIDataSourceManager(const IGIDataSourceManager&); // Purposefully not implemented.
  IGIDataSourceManager& operator=(const IGIDataSourceManager&); // Purposefully not implemented.

private:

}; // end class;

} // end namespace

#endif
