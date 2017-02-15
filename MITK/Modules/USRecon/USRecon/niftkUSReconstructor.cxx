/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "niftkUSReconstructor.h"
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
class USReconstructorPrivate {

public:

  USReconstructorPrivate();
  virtual ~USReconstructorPrivate();

  mitk::Image::Pointer DoReconstruction();
  void ClearData();
  void AddPair(mitk::Image::Pointer image,
               niftk::CoordinateAxesData::Pointer transform);

private:

  std::vector<std::pair<mitk::Image::Pointer, niftk::CoordinateAxesData::Pointer> > m_Data;
};


//-----------------------------------------------------------------------------
USReconstructorPrivate::USReconstructorPrivate()
{
}


//-----------------------------------------------------------------------------
USReconstructorPrivate::~USReconstructorPrivate()
{
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer USReconstructorPrivate::DoReconstruction()
{
  MITK_INFO << "USReconstructorPrivate: Doing Ultrasound Reconstruction with "
            << m_Data.size() << " samples.";

  // This just creates a dummy image.
  // This should create a new image, and fill it with reconstructed data.
  mitk::PixelType pt = mitk::MakeScalarPixelType<unsigned char>();
  mitk::Image::Pointer op = mitk::Image::New();
  unsigned int dim[] = { 5, 5, 5 };
  op->Initialize( pt, 3, dim);

  // And returns the image.
  return op;
}


//-----------------------------------------------------------------------------
void USReconstructorPrivate::ClearData()
{
  m_Data.clear();

  MITK_INFO << "USReconstructorPrivate: Cleared data down.";
}


//-----------------------------------------------------------------------------
void USReconstructorPrivate::AddPair(mitk::Image::Pointer image,
                                     niftk::CoordinateAxesData::Pointer transform)
{
  m_Data.push_back(std::make_pair<
                   mitk::Image::Pointer, niftk::CoordinateAxesData::Pointer>(image->Clone(),
                                                                             transform->Clone()));

  MITK_INFO << "USReconstructorPrivate: Collected snapshot:" << m_Data.size();
}


//-----------------------------------------------------------------------------
USReconstructor::USReconstructor()
: m_Impl(new USReconstructorPrivate())
{
}


//-----------------------------------------------------------------------------
USReconstructor::~USReconstructor()
{
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer USReconstructor::DoReconstruction()
{
  return m_Impl->DoReconstruction();
}


//-----------------------------------------------------------------------------
void USReconstructor::ClearData()
{
  m_Impl->ClearData();
}


//-----------------------------------------------------------------------------
void USReconstructor::AddPair(mitk::Image::Pointer image,
                              niftk::CoordinateAxesData::Pointer transform)
{
  m_Impl->AddPair(image, transform);
}

} // end namespace
