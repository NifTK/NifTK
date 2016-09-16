/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkBKMedicalDataSourceActivator_h
#define niftkBKMedicalDataSourceActivator_h

#include "niftkBKMedicalDataSourceFactory.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* \class BKMedicalDataSourceActivator
* \brief Activator to register the BKMedicalDataSourceFactory.
*/
class BKMedicalDataSourceActivator : public us::ModuleActivator
{
public:

  BKMedicalDataSourceActivator();
  ~BKMedicalDataSourceActivator();

  void Load(us::ModuleContext* context);
  void Unload(us::ModuleContext*);

private:

  BKMedicalDataSourceActivator(const BKMedicalDataSourceActivator&); // deliberately not implemented
  BKMedicalDataSourceActivator& operator=(const BKMedicalDataSourceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::BKMedicalDataSourceFactory> m_Factory;
};

} // end namespace

#endif
