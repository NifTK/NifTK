/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkToolFactoryMacros_h
#define __niftkToolFactoryMacros_h

// Note:
// MITK_TOOL_MACRO assumes that the type is in the 'mitk' namespace.
// To overcome this assumption, we create an 'alias' to our class with
// the desired name.

#define NIFTK_TOOL_MACRO(EXPORT_SPEC, TOOL_CLASS_NAME, DESCRIPTION)\
namespace mitk\
{\
class TOOL_CLASS_NAME : public niftk::TOOL_CLASS_NAME\
{\
};\
\
MITK_TOOL_MACRO(EXPORT_SPEC, TOOL_CLASS_NAME, DESCRIPTION);\
}\

// Note:
// QmitkToolSelectionBox looks for tool controls with a "Qmitk" prefix and
// "GUI" suffix added to the class name of the tool. To overcome this assumption
// we create an 'alias' to our class with the desired name.
// This version of the macro does not export the class.

#define FAKE_EXPORT_SPEC

#define NIFTK_TOOL_GUI_MACRO_NO_EXPORT(TOOL_CLASS_NAME, TOOL_GUI_CLASS_NAME, DESCRIPTION)\
\
class Qmitk ## TOOL_CLASS_NAME ## GUI : public TOOL_GUI_CLASS_NAME\
{\
};\
\
MITK_TOOL_GUI_MACRO(FAKE_EXPORT_SPEC, Qmitk ## TOOL_CLASS_NAME ## GUI, DESCRIPTION)\

// Note:
// QmitkToolSelectionBox looks for tool controls with a "Qmitk" prefix and
// "GUI" suffix added to the class name of the tool. To overcome this assumption
// we create an 'alias' to our class with the desired name.
// This version of the macro exports the class.

#define NIFTK_TOOL_GUI_MACRO(EXPORT_SPEC, TOOL_CLASS_NAME, TOOL_GUI_CLASS_NAME, DESCRIPTION)\
\
class EXPORT_SPEC Qmitk ## TOOL_CLASS_NAME ## GUI : public TOOL_GUI_CLASS_NAME\
{\
};\
\
MITK_TOOL_GUI_MACRO(EXPORT_SPEC, Qmitk ## TOOL_CLASS_NAME ## GUI, DESCRIPTION)\

#endif
