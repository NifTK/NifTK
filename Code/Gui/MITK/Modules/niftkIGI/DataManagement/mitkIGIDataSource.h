/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKIGIDATASOURCE_H
#define MITKIGIDATASOURCE_H

#include "niftkIGIExports.h"
#include <mitkDataStorage.h>
#include <mitkMessage.h>
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <list>
#include "mitkIGIDataType.h"

namespace mitk {

/**
 * \class IGIDataSource
 * \brief Base class for IGI Data Sources, where Data Sources can provide video data,
 * ultrasound data, tracker data etc.
 */
class NIFTKIGI_EXPORT IGIDataSource : public itk::Object
{

public:

  mitkClassMacro(IGIDataSource, itk::Object);

  /**
   * \brief Each tool should signal when the status has updated,
   * so for example the GUI can redraw, passing its internal identifier.
   */
  Message1<int> DataSourceStatusUpdated;

  /**
   * \brief Indicate to all listeners that this object has changed whether or not it is saving messages.
   */
  Message<> SaveStateChanged;

  /**
   * \brief Sets the identifier, which is just a tag to identify the tool by (i.e. item in a list).
   */
  itkSetMacro(Identifier, int);
  itkGetMacro(Identifier, int);

  /**
   * \brief Sets a name, useful for display purposes.
   */
  itkSetMacro(Name, std::string);
  itkGetMacro(Name, std::string);

  /**
   * \brief Sets a type, useful for display purposes.
   */
  itkSetMacro(Type, std::string);
  itkGetMacro(Type, std::string);

  /**
   * \brief Sets a status message, useful for display purposes.
   */
  itkSetMacro(Status, std::string);
  itkGetMacro(Status, std::string);

  /**
   * \brief Sets a description, useful for display purposes.
   */
  itkSetMacro(Description, std::string);
  itkGetMacro(Description, std::string);

  /**
   * \brief Sets the file name prefix, for where to save data.
   */
  itkSetMacro(SavePrefix, std::string);
  itkGetMacro(SavePrefix, std::string);

  /**
   * \brief Sets the tolerance for checking data.
   */
  itkSetMacro(TimeStampTolerance, mitk::IGIDataType::NifTKTimeStampType);
  itkGetMacro(TimeStampTolerance, mitk::IGIDataType::NifTKTimeStampType);

  /**
   * \brief Sets the data storage.
   */
  itkSetObjectMacro(DataStorage, mitk::DataStorage);
  itkGetConstMacro(DataStorage, mitk::DataStorage*);

  /**
   * \brief Returns true if we are saving messages and false otherwise.
   */
  itkGetMacro(SavingMessages, bool);
  void SetSaveState(bool isSaving);

  /**
   * \brief FrameRate is calculated internally, and can be retrieved here, and units should be equivalent to frames per second.
   */
  itkGetMacro(FrameRate, float);

  /**
   * \brief Get the time stamp that we are interested in.
   */
  itkGetMacro(RequestedTimeStamp, mitk::IGIDataType::NifTKTimeStampType);

  /**
   * \brief This is calculated internally, and represents the time-stamp of the "current" message,
   * which may be before or after the  RequestedTimeStamp.
   */
  itkGetMacro(ActualTimeStamp, mitk::IGIDataType::NifTKTimeStampType);

  /**
   * \brief Returns the current data item, with no iterating, or updating.
   */
  itkGetMacro(ActualData, mitk::IGIDataType::Pointer);

  /**
   * \brief Tools can have an optional Initialize function to perform any setup after construction,
   * with this class providing a default, do-nothing implementation.
   */
  virtual void Initialize() {};

  /**
   * \brief Derived classes can update the frame rate, as they receive data, and the units should be in frames per second.
   */
  virtual void UpdateFrameRate();

  /**
   * \brief Clears the internal buffer, which means completely destroying all the contents.
   */
  void ClearBuffer();

  /**
   * \brief Iterates through the buffer, clearing data, up to either the current time-stamp,
   * or if we are saving data, until we hit a piece of data that has not yet been saved.
   */
  void CleanBuffer();

  /**
   * \brief Returns the number of items in the buffer to aid testing.
   */
  unsigned long int GetBufferSize() const;

  /**
   * \brief Returns the first time stamp, or 0 if the buffer is empty.
   */
  mitk::IGIDataType::NifTKTimeStampType GetFirstTimeStamp() const;

  /**
   * \brief Returns the last time stamp, or 0 if the buffer is empty.
   */
  mitk::IGIDataType::NifTKTimeStampType GetLastTimeStamp() const;

  /**
   * \brief Add data to the buffer, derived classes can decide to reject the data,
   * so returns true if added and false otherwise.
   */
  bool AddData(mitk::IGIDataType::Pointer data);

  /**
   * \brief Processes the data for a given timestamp, returning true if
   * the current data is processed successfully and within time tolerance.
   */
  bool ProcessData(mitk::IGIDataType::NifTKTimeStampType requestedTimeStamp);

  /**
   * \brief Returns true if the current data frame is within time tolerances and false otherwise.
   */
  bool IsCurrentWithinTimeTolerance() const;

protected:

  IGIDataSource(); // Purposefully hidden.
  virtual ~IGIDataSource(); // Purposefully hidden.

  IGIDataSource(const IGIDataSource&); // Purposefully not implemented.
  IGIDataSource& operator=(const IGIDataSource&); // Purposefully not implemented.

  /**
   * \brief Derived classes should implement this to decide if they can handle a certain type of data,
   * returning true if it can be handled, and false otherwise.
   */
  virtual bool CanHandleData(mitk::IGIDataType::Pointer data) const = 0;

  /**
   * \brief Will iterate through the internal buffer, returning the
   * closest message to the requested time stamp, and sets the
   * ActualTimeStamp accordingly, or else return NULL if it can't be found.
   */
  virtual mitk::IGIDataType::Pointer RequestData(mitk::IGIDataType::NifTKTimeStampType requestedTimeStamp);

  /**
   * \brief Derived classes implement this to provide some kind of update based on the given data,
   * which must be NOT NULL, and should return true for successful and false otherwise.
   */
  virtual bool Update(mitk::IGIDataType::Pointer data) { return true; }

private:

  mitk::DataStorage                              *m_DataStorage;
  int                                             m_Identifier;
  float                                           m_FrameRate;
  std::string                                     m_Name;
  std::string                                     m_Type;
  std::string                                     m_Status;
  std::string                                     m_Description;
  bool                                            m_SavingMessages;
  std::string                                     m_SavePrefix;
  std::list<mitk::IGIDataType::Pointer>           m_Buffer;
  std::list<mitk::IGIDataType::Pointer>::iterator m_BufferIterator;
  mitk::IGIDataType::NifTKTimeStampType           m_RequestedTimeStamp;
  mitk::IGIDataType::NifTKTimeStampType           m_ActualTimeStamp;
  mitk::IGIDataType::NifTKTimeStampType           m_TimeStampTolerance;
  mitk::IGIDataType::Pointer                      m_ActualData;

}; // end class

} // end namespace

#endif

