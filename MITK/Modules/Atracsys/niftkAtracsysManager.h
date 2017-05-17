/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkAtracsysManager_h
#define niftkAtracsysManager_h

#include "niftkAtracsysExports.h"

#include <QObject>

namespace niftk
{

/**
 * \class AtracsysManager
 * \brief Interface to Atracsys tracker that can be used
 * from either command line or GUI clients within NifTK.
 */
class NIFTKATRACSYS_EXPORT AtracsysManager : public QObject
{

  Q_OBJECT

public:

  AtracsysManager();
  virtual ~AtracsysManager();

}; // end class

} // end namespace

#endif
