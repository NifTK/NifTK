/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPointRegServiceUsingSVD_h
#define niftkPointRegServiceUsingSVD_h

#include <niftkPointRegServiceI.h>

namespace niftk
{

class PointRegServiceUsingSVD : public niftk::PointRegServiceI
{
public:

  PointRegServiceUsingSVD();
  ~PointRegServiceUsingSVD();

  virtual double PointBasedRegistration(const mitk::PointSet::Pointer& fixedPoints,
                                        const mitk::PointSet::Pointer& movingPoints,
                                        vtkMatrix4x4& matrix) const;

private:

  PointRegServiceUsingSVD(const PointRegServiceUsingSVD&); // deliberately not implemented
  PointRegServiceUsingSVD& operator=(const PointRegServiceUsingSVD&); // deliberately not implemented

};

} // end namespace

#endif

