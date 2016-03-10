/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGITrackerDataType_h
#define niftkIGITrackerDataType_h

#include <niftkIGIDataType.h>

#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

namespace niftk
{

/**
 * \class IGITrackerDataType
 * \brief Class to represent tracker data.
 *
 * (normally NDI Aurora, Spectra, Polaris etc).
 */
class IGITrackerDataType : public IGIDataType
{
public:

  mitkClassMacro(IGITrackerDataType, IGIDataType);
  itkNewMacro(IGITrackerDataType);

  void SetTrackingData(vtkSmartPointer<vtkMatrix4x4> data);
  vtkSmartPointer<vtkMatrix4x4> GetTrackingData() const;

  itkSetStringMacro(ToolName);
  itkGetStringMacro(ToolName);

protected:

  IGITrackerDataType(); // Purposefully hidden.
  virtual ~IGITrackerDataType(); // Purposefully hidden.

  IGITrackerDataType(const IGITrackerDataType&); // Purposefully not implemented.
  IGITrackerDataType& operator=(const IGITrackerDataType&); // Purposefully not implemented.

private:

  std::string                   m_ToolName;
  vtkSmartPointer<vtkMatrix4x4> m_TrackingData;
};

} // end namespace

#endif
