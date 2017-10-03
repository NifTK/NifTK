/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPosnTool_h
#define niftkPosnTool_h

#include "niftkMIDASExports.h"

#include "niftkTool.h"

namespace niftk
{

/**
 * \class PosnTool
 * \brief Dummy class, as the MIDAS posn tool, just enables you to change the
 * position of the slices in 2 or 3 windows, which is the default behaviour of
 * the MITK ortho-viewer anyway.
 */
class NIFTKMIDAS_EXPORT PosnTool : public Tool
{

public:
  mitkClassMacro(PosnTool, Tool)
  itkNewMacro(PosnTool)

  virtual const char* GetName() const override;

  virtual const char** GetXPM() const override;

protected:

  PosnTool(); // purposefully hidden
  virtual ~PosnTool(); // purposefully hidden

private:

};

}

#endif
