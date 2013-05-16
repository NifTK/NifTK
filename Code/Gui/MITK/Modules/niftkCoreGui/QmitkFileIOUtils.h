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
 * \brief Save the matrix to a plain text file of 4 rows of 4 space separated numbers.
 * \param fileName full path of file name
 * \param matrix a matrix
 * \param bool true if successful and false otherwise
 */
NIFTKCOREGUI_EXPORT vtkMatrix4x4* LoadMatrix4x4FromFile(const QString &fileName);

/**
 * \brief Save the matrix to a plain text file of 4 rows of 4 space separated numbers.
 * \param fileName full path of file name
 * \param matrix a matrix
 */
NIFTKCOREGUI_EXPORT bool SaveMatrix4x4ToFile (const QString& fileName, const vtkMatrix4x4& matrix);

#endif // QmitkFileIOUtil_h
