/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGITrackerBackend_h
#define niftkIGITrackerBackend_h

#include <niftkIGITrackersExports.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkDataStorage.h>

namespace niftk
{
class NIFTKIGITRACKERS_EXPORT IGITrackerBackend : public itk::Object
{
public:

  mitkClassMacroItkParent(IGITrackerBackend, itk::Object)
  itkGetMacro(Lag, int);
  itkSetMacro(Lag, int);

protected:

  IGITrackerBackend(mitk::DataStorage::Pointer dataStorage); // Purposefully hidden.
  virtual ~IGITrackerBackend(); // Purposefully hidden.

  IGITrackerBackend(const IGITrackerBackend&); // Purposefully not implemented.
  IGITrackerBackend& operator=(const IGITrackerBackend&); // Purposefully not implemented.

  mitk::DataStorage::Pointer m_DataStorage;
  int                        m_Lag;
};

} // end namespace

#endif
