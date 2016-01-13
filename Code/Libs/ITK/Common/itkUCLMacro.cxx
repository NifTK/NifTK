/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkUCLMacro.h"

#include <itkTextOutput.h>

namespace niftk
{

KeepTextOutputInShell::KeepTextOutputInShell()
{
  itk::OutputWindow::SetInstance(itk::TextOutput::New());
}

KeepTextOutputInShell KeepTextOutputInShell::s_KeepTextOutputInShell;

}
