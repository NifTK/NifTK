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

#ifndef MITKMIDASDATASTORAGEPROPERTYLISTENER_H_
#define MITKMIDASDATASTORAGEPROPERTYLISTENER_H_

#include "niftkMitkExtExports.h"

#include <itkObject.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>

namespace mitk
{

/**
 * \class DataStoragePropertyListener
 * \brief Base class for objects that Listen to data storage for a specific property such as "visibility".
 *
 * This class is derived from itk::Object so we can use things like the ITK setter/getter macros, listening to
 * Modified events via the Observer pattern etc.
 *
 * Derived classes must implement OnPropertyChanged.
 */
class NIFTKMITKEXT_EXPORT DataStoragePropertyListener : public itk::Object
{

public:

  mitkClassMacro(DataStoragePropertyListener, itk::Object);
  itkNewMacro(DataStoragePropertyListener);
  mitkNewMacro1Param(DataStoragePropertyListener, const mitk::DataStorage::Pointer);

  /// \brief Get the data storage.
  itkGetMacro(DataStorage, mitk::DataStorage::Pointer);

  /// \brief Set the data storage.
  void SetDataStorage(const mitk::DataStorage::Pointer dataStorage);

  /// \brief Set/Get the property name.
  itkSetMacro(PropertyName, std::string);
  itkGetMacro(PropertyName, std::string);

protected:

  DataStoragePropertyListener();
  DataStoragePropertyListener(const mitk::DataStorage::Pointer);
  virtual ~DataStoragePropertyListener();

  DataStoragePropertyListener(const DataStoragePropertyListener&); // Purposefully not implemented.
  DataStoragePropertyListener& operator=(const DataStoragePropertyListener&); // Purposefully not implemented.

  /// \brief In this class, we do nothing, as subclasses should re-define this.
  virtual void OnPropertyChanged(const itk::EventObject&) {};

  /// \brief Will refresh the observers of the named property, and sub-classes should call this at the appropriate time.
  virtual void UpdateObserverToPropertyMap();

  /// \brief Will remove all observers from the m_ObserverToPropertyMap, and sub-classes should call this at the appropriate time.
  virtual void RemoveAllFromObserverToPropertyMap();

private:

  /// \brief Called to register to the data storage.
  void Activate(const mitk::DataStorage::Pointer storage);

  /// \brief Called to un-register from the data storage.
  void Deactivate();

  // We observe all the global properties for each registered node.
  typedef std::map<unsigned long, mitk::BaseProperty::Pointer> ObserverToPropertyMap;
  ObserverToPropertyMap m_ObserverToPropertyMap;

  /// \brief This object MUST be connected to a datastorage for it to work.
  mitk::DataStorage::Pointer m_DataStorage;

  /// \brief The name of the property we are tracking.
  std::string m_PropertyName;
};

} // end namespace

#endif
