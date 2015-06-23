/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkICPRegService_h
#define niftkICPRegService_h

#include <niftkSurfaceRegServiceI.h>

namespace niftk
{

/**
* @class ICPRegService
* @brief Implements niftk::SurfaceRegServiceI using niftk::ICPBasedRegistration.
*/
class ICPRegService : public niftk::SurfaceRegServiceI
{
public:

  ICPRegService();
  ~ICPRegService();

  /**
  * \see niftk::SurfaceRegServiceI
  * \throws mitk::Exception for all errors
  */
  virtual double SurfaceBasedRegistration(const mitk::DataNode::Pointer& fixedDataSet,
                                          const mitk::DataNode::Pointer& movingDataSet,
                                          vtkMatrix4x4& matrix) const;

private:

  ICPRegService(const ICPRegService&); // deliberately not implemented
  ICPRegService& operator=(const ICPRegService&); // deliberately not implemented

};

} // end namespace

#endif
