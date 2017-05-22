/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkAtracsysManager.h"
#include <mitkLogMacros.h>
#include <mitkExceptionMacro.h>
#include <ftkInterface.h>
#include <helpers.hpp>

namespace niftk
{

//-----------------------------------------------------------------------------
class AtracsysManagerPrivate
{
  Q_DECLARE_PUBLIC(AtracsysManager)
  AtracsysManager* const q_ptr;

public:

  AtracsysManagerPrivate(AtracsysManager* q);
  ~AtracsysManagerPrivate();

private:

  void CheckError(ftkLibrary lib);

  ftkLibrary m_Lib;
};


//-----------------------------------------------------------------------------
AtracsysManagerPrivate::AtracsysManagerPrivate(AtracsysManager* q)
: q_ptr(q)
, m_Lib(nullptr)
{

  m_Lib = ftkInit();
  if ( m_Lib == nullptr )
  {
    mitkThrow() << "Cannot initialize Atracsys driver.";
  }

  DeviceData device;
  device.SerialNumber = 0uLL;

  ftkError err = ftkEnumerateDevices( m_Lib, fusionTrackEnumerator, &device );
  if ( err != FTK_OK )
  {
    this->CheckError(m_Lib);
  }

  if ( device.SerialNumber == 0uLL )
  {
    mitkThrow() << "No Atracsys device connected.";
  }

  uint64 sn( device.SerialNumber );
  MITK_INFO << "Connected to Atracsys SN:" << sn;
}


//-----------------------------------------------------------------------------
AtracsysManagerPrivate::~AtracsysManagerPrivate()
{
  Q_Q(AtracsysManager);

  if (m_Lib != nullptr)
  {
    ftkClose( &m_Lib );
  }
}


//-----------------------------------------------------------------------------
void AtracsysManagerPrivate::CheckError(ftkLibrary lib)
{
  ftkErrorExt ext;
  ftkError err = ftkGetLastError( lib, &ext );
  if ( err == FTK_OK ) // means we successfully retrieved the error message.
  {
    std::string message;
    if ( ext.isError() )
    {
      ext.errorString( message );
      MITK_ERROR << "AtracsysManagerPrivate:" << message;
      mitkThrow() << message;
    }
    if ( ext.isWarning() )
    {
      ext.warningString( message );
      MITK_WARN << "AtracsysManagerPrivate:" << message;
    }
    ext.messageStack( message );
    if ( message.size() > 0u )
    {
      MITK_INFO << "AtracsysManagerPrivate:Stack:\n" << message;
    }
  }
}


//-----------------------------------------------------------------------------
AtracsysManager::AtracsysManager()
: d_ptr(new AtracsysManagerPrivate(this))
{
  MITK_INFO << "Creating AtracsysManager";
}


//-----------------------------------------------------------------------------
AtracsysManager::~AtracsysManager()
{
}

} // end namespace
