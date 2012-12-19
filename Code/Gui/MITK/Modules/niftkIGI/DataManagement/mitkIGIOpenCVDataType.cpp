/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkIGIOpenCVDataType.h"

namespace mitk
{

//-----------------------------------------------------------------------------
IGIOpenCVDataType::IGIOpenCVDataType()
: m_Image(NULL)
{
}

//-----------------------------------------------------------------------------
IGIOpenCVDataType::~IGIOpenCVDataType()
{
  if (m_Image != NULL)
  {
    delete m_Image;
  }
}


//-----------------------------------------------------------------------------
void IGIOpenCVDataType::CloneImage(const IplImage *image)
{
  if (m_Image != NULL)
  {
    delete m_Image;
  }
  m_Image = cvCloneImage(image);
}

} // end namespace

