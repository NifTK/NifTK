/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkFileIOUtils.h"
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <vtkFunctions.h>

//-----------------------------------------------------------------------------
vtkMatrix4x4* LoadMatrix4x4FromFile(const QString &fileName)
{
  return LoadMatrix4x4FromFile(fileName.toStdString(), true);
}


//-----------------------------------------------------------------------------
bool SaveMatrix4x4ToFile (const QString& fileName, const vtkMatrix4x4& matrix)
{
  return SaveMatrix4x4ToFile(fileName, matrix);
}
