/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkLabeledLookupTableProperty_h
#define mitkLabeledLookupTableProperty_h

#include "niftkCoreExports.h"
#include "mitkNamedLookupTableProperty.h"

namespace mitk {

/**
 * \class LabeledLookupTableProperty
 * \brief Provides a property so that each value/color has an associated name.
 * \ingroup uk_ac_ucl_cmic_imagelookuptables_internal
 */
class NIFTKCORE_EXPORT LabeledLookupTableProperty: public NamedLookupTableProperty
{

public:
  typedef std::pair<int, std::string> LabelType;
  typedef std::vector<LabelType> LabelsListType;

  mitkClassMacro(LabeledLookupTableProperty, NamedLookupTableProperty);
  itkNewMacro(LabeledLookupTableProperty);
  mitkNewMacro3Param(LabeledLookupTableProperty, 
                     const std::string&, 
					           const mitk::LookupTable::Pointer, 
					           LabelsListType);
  mitkNewMacro4Param(LabeledLookupTableProperty, 
                     const std::string&, 
					           const mitk::LookupTable::Pointer, 
					           LabelsListType, 
					           bool);

  /** Get/set list of labels*/
  LabelsListType GetLabels(){return m_Labels;};
  void SetLabels(LabelsListType labels){m_Labels = labels;};

protected:

  virtual ~LabeledLookupTableProperty();
  LabeledLookupTableProperty();
  LabeledLookupTableProperty(const LabeledLookupTableProperty& other);
  LabeledLookupTableProperty(const std::string& name, 
                             const mitk::LookupTable::Pointer lut, 
							               LabelsListType labels);
  LabeledLookupTableProperty(const std::string& name, 
                             const mitk::LookupTable::Pointer lut, 
							               LabelsListType labels, 
							               bool scale);

private:

  LabeledLookupTableProperty& operator=(const LabeledLookupTableProperty&); // Purposefully not implemented
  itk::LightObject::Pointer InternalClone() const;

  virtual bool IsEqual(const BaseProperty& property) const;
  virtual bool Assign(const BaseProperty& property);

  LabelsListType m_Labels;
};

} // namespace mitk


#endif