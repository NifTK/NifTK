/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef CoordinateAxesDataOpUpdate_h
#define CoordinateAxesDataOpUpdate_h

#include "niftkCoreExports.h"
#include <mitkOperation.h>
#include <mitkOperationActor.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

namespace mitk
{

class NIFTKCORE_EXPORT CoordinateAxesDataOpUpdate : public mitk::Operation
{
public:
  CoordinateAxesDataOpUpdate(
      mitk::OperationType type,
      const vtkMatrix4x4& matrix,
      const std::string &nodeName
      );
  ~CoordinateAxesDataOpUpdate();

  vtkSmartPointer<vtkMatrix4x4> GetMatrix() const;
  std::string GetNodeName() const { return m_NodeName; }

private:
  vtkSmartPointer<vtkMatrix4x4> m_Matrix;
  std::string m_NodeName;
};

} // end namespace

#endif // CoordinateAxesDataOpUpdate_h
