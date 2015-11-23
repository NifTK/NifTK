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
#include <QString.h>

namespace mitk
{

/**
 * \class LabeledLookupTableProperty
 * \brief Provides a property so that each value/color has an associated name.
 * \ingroup uk_ac_ucl_cmic_imagelookuptables_internal
 */
class NIFTKCORE_EXPORT LabeledLookupTableProperty : public NamedLookupTableProperty
{
public:

  typedef std::pair<int, QString> LabelType;
  typedef std::vector<LabelType> LabelListType;

  mitkClassMacro(LabeledLookupTableProperty, NamedLookupTableProperty);
  itkNewMacro(LabeledLookupTableProperty);
  mitkNewMacro3Param(LabeledLookupTableProperty, 
                     const std::string&, 
                     const mitk::LookupTable::Pointer, 
                     const LabelListType&);
  mitkNewMacro4Param(LabeledLookupTableProperty, 
                     const std::string&, 
                     const mitk::LookupTable::Pointer, 
                     const LabelListType&, 
                     bool);

  /** Get list of labels*/
  inline LabelListType GetLabels() const {return m_Labels;}

  /** Set list of labels*/
  inline void SetLabels(const LabelListType& labels){m_Labels = labels;}

protected:

  virtual ~LabeledLookupTableProperty();
  LabeledLookupTableProperty();
  LabeledLookupTableProperty(const LabeledLookupTableProperty& other);
  LabeledLookupTableProperty(const std::string& name, 
                             const mitk::LookupTable::Pointer lut, 
                             const LabelListType& labels);
  LabeledLookupTableProperty(const std::string& name, 
                             const mitk::LookupTable::Pointer lut, 
                             const LabelListType& labels, 
                             bool scale);

private:

  LabeledLookupTableProperty& operator=(const LabeledLookupTableProperty&); // Purposefully not implemented
  itk::LightObject::Pointer InternalClone() const;

  virtual bool IsEqual(const BaseProperty& property) const;
  virtual bool Assign(const BaseProperty& property);

  LabelListType m_Labels;
};

} // namespace mitk


#endif
