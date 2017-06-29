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
#include <mitkVector.h>

namespace niftk
{

/**
 * \class IGITrackerDataType
 * \brief Class to represent tracker data.
 *
 * (normally NDI Aurora, Spectra, Vicra etc).
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

  void SetTransform(const mitk::Point4D& rotation, const mitk::Vector3D& translation);
  void GetTransform(mitk::Point4D& rotation, mitk::Vector3D& translation) const;

  /**
  * \brief Overrides base class, but only copies IGITrackerDataType.
  */
  virtual void Clone(const IGIDataType&) override;

private:

  std::string    m_ToolName;
  mitk::Point4D  m_Rotation;
  mitk::Vector3D m_Translation;
};

} // end namespace

#endif
