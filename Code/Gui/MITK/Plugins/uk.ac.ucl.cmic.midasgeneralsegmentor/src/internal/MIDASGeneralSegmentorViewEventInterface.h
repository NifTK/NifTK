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

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _MIDASGENERALSEGMENTORVIEWEVENTINTERFACE_H_INCLUDED
#define _MIDASGENERALSEGMENTORVIEWEVENTINTERFACE_H_INCLUDED

#include "itkObject.h"
#include "mitkOperationActor.h"

class MIDASGeneralSegmentorView;

/**
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 *
 * \class MIDASGeneralSegmentorViewEventInterface
 * \brief Interface class, simply to callback onto MIDASGeneralSegmentorView class for Undo/Redo purposes.
 */
class MIDASGeneralSegmentorViewEventInterface: public itk::Object, public mitk::OperationActor
{
public:
  MIDASGeneralSegmentorViewEventInterface();
  ~MIDASGeneralSegmentorViewEventInterface();
  void SetMIDASGeneralSegmentorView( MIDASGeneralSegmentorView* view );
  virtual void  ExecuteOperation(mitk::Operation* op);
private:
  MIDASGeneralSegmentorView* m_View;
};

#endif
