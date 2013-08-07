/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkFileIOUtils.h>
#include <niftkVTKFunctions.h>

namespace mitk {

//-----------------------------------------------------------------------------
vtkMatrix4x4* LoadVtkMatrix4x4FromFile(const std::string& fileName)
{
  return niftk::LoadMatrix4x4FromFile(fileName, true);
}


//-----------------------------------------------------------------------------
bool SaveVtkMatrix4x4ToFile (const std::string& fileName, const vtkMatrix4x4& matrix)
{
  bool isSuccessful = false;
  if (fileName.length() > 0)
  {
    isSuccessful = niftk::SaveMatrix4x4ToFile(fileName, matrix);
  }
  return isSuccessful;
}


} // end namespace
