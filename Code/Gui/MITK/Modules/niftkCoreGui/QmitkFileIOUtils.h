/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkFileIOUtils_h
#define QmitkFileIOUtils_h

#include "niftkCoreGuiExports.h"
#include <vtkMatrix4x4.h>
#include <QString>

/**
 * \file QmitkFileIOUtils.h
 * \brief Various file IO stuff, like loading transformations from file, that also use Qt.
 */

/**
 * \brief Loads a plain 4x4 matrix from a text file.
 * \return a new vtkMatrix4x4 which becomes the responsibility of the caller, or return NULL if it fails.
 */
NIFTKCOREGUI_EXPORT vtkMatrix4x4* Load4x4MatrixFromFile(const QString &fileName);

#endif // QmitkFileIOUtil_h
