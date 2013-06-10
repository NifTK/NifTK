/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef PointSetUpdate_h
#define PointSetUpdate_h

#include "niftkCoreExports.h"
#include <mitkOperation.h>
#include <mitkPointSet.h>
#include <mitkVector.h>

namespace mitk {

/**
 * \brief Operation class to enable updating an mitk::PointSet with Undo/Redo.
 */
class NIFTKCORE_EXPORT PointSetUpdate : public mitk::Operation
{
public:

  PointSetUpdate(
      mitk::OperationType type,
      mitk::PointSet::Pointer pointSet
      );
  virtual ~PointSetUpdate();

  const mitk::PointSet* GetPointSet() const { return m_PointSet.GetPointer(); }
  void Clear() { m_PointSet->Clear(); }
  void AppendPoint(const mitk::Point3D& point);

private:
  mitk::PointSet::Pointer m_PointSet;
};

} // end namespace

#endif // PointSetUpdate_h
