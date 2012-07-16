/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : Miklos Espak (espakm@gmail.com)

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/


#ifndef NamedLookupTableProperty_h
#define NamedLookupTableProperty_h

#include "niftkMitkExtExports.h"
#include "mitkLookupTableProperty.h"

namespace mitk {

/**
 * \class NamedLookupTableProperty
 * \brief Provides a property so we can see the lookup table name in the property window.
 * \ingroup uk_ac_ucl_cmic_imagelookuptables_internal
 */
class NIFTKMITKEXT_EXPORT NamedLookupTableProperty : public mitk::LookupTableProperty
{

protected:
  NamedLookupTableProperty();

  NamedLookupTableProperty(const std::string& name, const mitk::LookupTable::Pointer lut);

public:
  mitkClassMacro(NamedLookupTableProperty, mitk::LookupTableProperty);

  itkNewMacro(NamedLookupTableProperty);
  mitkNewMacro2Param(NamedLookupTableProperty, const std::string&, const mitk::LookupTable::Pointer);

  virtual ~NamedLookupTableProperty();

  virtual std::string GetValueAsString() const;

private:
  std::string m_Name;
};

} // namespace mitk

#endif /* NamedLookupTableProperty_h */
