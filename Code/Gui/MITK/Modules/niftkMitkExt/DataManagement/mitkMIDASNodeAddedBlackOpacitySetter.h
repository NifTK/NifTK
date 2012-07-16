/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKMIDASNODEADDEDBLACKOPACITYSETTER_H
#define MITKMIDASNODEADDEDBLACKOPACITYSETTER_H

#include "niftkMitkExtExports.h"
#include <mitkDataStorage.h>
#include "mitkDataStorageListener.h"
#include "mitkMIDASEnums.h"

namespace mitk
{

class DataNode;

/**
 * \class MIDASNodeAddedBlackOpacitySetter
 * \brief When a node is added to data storage, will set the black opacity to 1.
 */
class NIFTKMITKEXT_EXPORT MIDASNodeAddedBlackOpacitySetter : public DataStorageListener
{
public:

  mitkClassMacro(MIDASNodeAddedBlackOpacitySetter, DataStorageListener);
  itkNewMacro(MIDASNodeAddedBlackOpacitySetter);
  mitkNewMacro1Param(MIDASNodeAddedBlackOpacitySetter, const mitk::DataStorage::Pointer);

protected:

  MIDASNodeAddedBlackOpacitySetter();
  MIDASNodeAddedBlackOpacitySetter(const mitk::DataStorage::Pointer);
  virtual ~MIDASNodeAddedBlackOpacitySetter();

  MIDASNodeAddedBlackOpacitySetter(const MIDASNodeAddedBlackOpacitySetter&); // Purposefully not implemented.
  MIDASNodeAddedBlackOpacitySetter& operator=(const MIDASNodeAddedBlackOpacitySetter&); // Purposefully not implemented.

  /// \brief Called when a DataStorage AddNodeEvent was emmitted and may be reimplemented by deriving classes.
  virtual void NodeAdded(mitk::DataNode* node);

private:

};

} // end namespace

#endif
