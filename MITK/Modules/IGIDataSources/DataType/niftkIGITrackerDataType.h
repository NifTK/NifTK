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

#include "niftkIGIDataSourcesExports.h"
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
class NIFTKIGIDATASOURCES_EXPORT IGITrackerDataType : public IGIDataType
{
public:

  IGITrackerDataType();
  virtual ~IGITrackerDataType();

  IGITrackerDataType(const IGITrackerDataType&);             // Copy constructor
  IGITrackerDataType& operator=(const IGITrackerDataType&);  // Copy assignment
  IGITrackerDataType(IGITrackerDataType&&);                  // Move constructor
  IGITrackerDataType& operator=(IGITrackerDataType&&);       // Move assignment

  void SetToolName(const std::string& toolName) { m_ToolName = toolName; }
  std::string GetToolName() const { return m_ToolName; }

  void SetTrackingMatrix(const vtkSmartPointer<vtkMatrix4x4>& data);
  vtkSmartPointer<vtkMatrix4x4> GetTrackingMatrix() const;

  /**
   * \brief Sets the 4x4 matrix from a vector of 16 doubles.
   */
  void SetTrackingData(const std::vector<double>& transform);

private:

  std::string                   m_ToolName;
  vtkSmartPointer<vtkMatrix4x4> m_TrackingMatrix;
};

} // end namespace

#endif
