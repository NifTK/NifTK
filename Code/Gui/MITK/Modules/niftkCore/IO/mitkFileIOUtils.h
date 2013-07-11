/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkFileIOUtils_h
#define mitkFileIOUtils_h

#include "niftkCoreExports.h"
#include <vtkMatrix4x4.h>
#include <mitkImage.h>

namespace mitk {

/**
 * \file mitkFileIOUtils.h
 * \brief Various file IO stuff, like loading transformations from file.
 */

/**
 * \brief Load a plain text file of 4 rows of 4 space separated numbers into a vtkMatrix4x4.  
 * \param fileName full path of file name
 * \return vtkMatrix4x4* that the caller is responsible for
 */
NIFTKCORE_EXPORT vtkMatrix4x4* LoadVtkMatrix4x4FromFile(const std::string &fileName);

/**
 * \brief Save the matrix to a plain text file of 4 rows of 4 space separated numbers.
 * \param fileName full path of file name
 * \param matrix a matrix
 * \return true if successful and false otherwise 
 */
NIFTKCORE_EXPORT bool SaveVtkMatrix4x4ToFile (const std::string& fileName, const vtkMatrix4x4& matrix);

/**
 * \brief Saves the given transformation to file.
 * \param[In] fileName the full absolute path of the file to be saved to, which if it already exists will be silently over-written.
 * \param[In] transform transformation matrix.
 * \return bool true if successful and false otherwise.
 */
NIFTKCORE_EXPORT bool SaveVtkMatrix4x4ToFileIfFileName(const std::string& fileName, const vtkMatrix4x4& transform);

} // end namespace

#endif // QmitkFileIOUtil_h
