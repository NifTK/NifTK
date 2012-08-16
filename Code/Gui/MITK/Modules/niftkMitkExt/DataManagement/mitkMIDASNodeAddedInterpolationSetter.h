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

#ifndef MITKMIDASNODEADDEDINTERPOLATIONSETTER_H
#define MITKMIDASNODEADDEDINTERPOLATIONSETTER_H

#include "niftkMitkExtExports.h"
#include <mitkDataStorage.h>
#include "mitkDataStorageListener.h"
#include "mitkMIDASEnums.h"

namespace mitk
{

class DataNode;

/**
 * \class MIDASNodeAddedInterpolationSetter
 * \brief When a node is added to data storage, will set initial interpolation properties.
 */
class NIFTKMITKEXT_EXPORT MIDASNodeAddedInterpolationSetter : public DataStorageListener
{
public:

  mitkClassMacro(MIDASNodeAddedInterpolationSetter, DataStorageListener);
  itkNewMacro(MIDASNodeAddedInterpolationSetter);
  mitkNewMacro1Param(MIDASNodeAddedInterpolationSetter, const mitk::DataStorage::Pointer);

  /// \brief Sets/Gets the default interpolation type,
  itkSetMacro(DefaultInterpolation, MIDASDefaultInterpolationType);
  itkGetMacro(DefaultInterpolation, MIDASDefaultInterpolationType);

protected:

  MIDASNodeAddedInterpolationSetter();
  MIDASNodeAddedInterpolationSetter(const mitk::DataStorage::Pointer);
  virtual ~MIDASNodeAddedInterpolationSetter();

  MIDASNodeAddedInterpolationSetter(const MIDASNodeAddedInterpolationSetter&); // Purposefully not implemented.
  MIDASNodeAddedInterpolationSetter& operator=(const MIDASNodeAddedInterpolationSetter&); // Purposefully not implemented.

  /**
   * \see DataStorageListener::NodeAdded
   */
  virtual void NodeAdded(mitk::DataNode* node);

private:

  /// \brief Keeps track of the default interpolation.
  MIDASDefaultInterpolationType m_DefaultInterpolation;
};

} // end namespace

#endif
