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

//-----------------------------------------------------------------------------
vtkMatrix4x4* Load4x4MatrixFromFile(const QString &fileName)
{
  vtkMatrix4x4 *result = NULL;

  QFile matrixFile(fileName);
  if (!matrixFile.open(QIODevice::ReadOnly | QIODevice::Text))
  {
    qDebug() << "Load4x4MatrixFromFile: failed to open file:" << fileName;
    return result;
  }

  QTextStream matrixIn(&matrixFile);

  result = vtkMatrix4x4::New();

  for ( int row = 0 ; row < 4 ; row ++ )
  {
    for ( int col = 0 ; col < 4 ; col ++ )
    {
      double tmp;
      matrixIn >> tmp;
      result->SetElement(row, col, tmp);
    }
  }
  matrixFile.close();

  return result;
}

