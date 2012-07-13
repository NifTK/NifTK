/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKMIDASINITIALPROPERTYMANAGER_H_
#define MITKMIDASINITIALPROPERTYMANAGER_H_

#include "niftkMitkExtExports.h"
#include <mitkDataStorage.h>
#include "mitkDataStorageListener.h"
#include "mitkMIDASEnums.h"

namespace mitk
{

class DataNode;

/**
 * \class MIDASInitialPropertyManager
 * \brief Contains the logic for what to set on a node when it is added to data storage.
 */
class NIFTKMITKEXT_EXPORT MIDASInitialPropertyManager : public DataStorageListener
{
public:

  /// \brief This class must (checked with assert) have a non-NULL mitk::DataStorage.
  MIDASInitialPropertyManager(mitk::DataStorage::Pointer dataStorage);

  /// \brief Destructor, which unregisters all the listeners.
  virtual ~MIDASInitialPropertyManager();

  /// \brief Sets the default interpolation type, which takes effect when a new image is dropped.
  void SetDefaultInterpolationType(MIDASDefaultInterpolationType interpolation) { m_DefaultInterpolation = interpolation; }

  /// \brief Returns the default interpolation type, which takes effect when a new image is dropped.
  MIDASDefaultInterpolationType GetDefaultInterpolationType() const { return m_DefaultInterpolation; }

protected:

  /// \brief Called when a DataStorage AddNodeEvent was emmitted and may be reimplemented by deriving classes.
  virtual void NodeAdded(mitk::DataNode* node);

private:

  // Keeps track of the default interpolation.
  MIDASDefaultInterpolationType m_DefaultInterpolation;
};

} // end namespace

#endif
