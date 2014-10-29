/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef PointSetCropperEventInterface_h
#define PointSetCropperEventInterface_h

#include "itkObject.h"

#include "mitkOperation.h"
#include "mitkOperationActor.h"

class QmitkPointSetCropper;

namespace mitk
{
  class PointSetCropperEventInterface : public itk::Object, public OperationActor
  {

  public:

    PointSetCropperEventInterface();
    ~PointSetCropperEventInterface();

    void SetPointSetCropper( QmitkPointSetCropper* PointSetCropper )
    {
      m_PointSetCropper = PointSetCropper;
    }

    virtual void  ExecuteOperation(mitk::Operation* op);

  private:

    QmitkPointSetCropper* m_PointSetCropper;

  };
}

#endif // PointSetCropperEventInterface_h
