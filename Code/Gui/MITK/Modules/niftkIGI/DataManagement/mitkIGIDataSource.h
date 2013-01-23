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
#include <NiftyLinkUtils.h>
#include <mitkDataStorage.h>
#include <mitkMessage.h>
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <itkFastMutexLock.h>
#include <list>
#include "mitkIGIDataType.h"

namespace mitk {

/**
 * \class IGIDataSource
 * \brief Base class for IGI Data Sources, where Data Sources can provide video data,
 * ultrasound data, tracker data etc.
 *
 * NOTE: All timestamps should be in UTC format. Also, take care NOT to expose a pointer to the
 * igtl::TimeStamp object. You should only ever expose a copy of this data, or an equivalent
 * representation of it, i.e. if you Set/Get the igtlUint64 values, then NO-ONE can modify the timestamp
 * and set the time to TAI for example.
 */
class NIFTKIGI_EXPORT IGIDataSource : public itk::Object
{

public:

  mitkClassMacro(IGIDataSource, itk::Object);

  /**
   * \brief Each tool should signal when the status has updated by
   * emitting its internal identiier, so that for example the GUI can redraw.
   */
  Message1<int> DataSourceStatusUpdated;

  /**
   * \brief Indicate to all listeners that this object has changed whether or not it is saving messages.
   */
  Message<> SaveStateChanged;

  /**
   * \brief Sets the identifier, which is just a tag to identify the tool by (i.e. item number in a list).
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
  * \brief A single source can have multiple tools attached
  */
  itkSetMacro(NumberOfTools, int);
  itkGetMacro(NumberOfTools, int);
  
  /**
   * \brief Sets the time tolerance for checking data, implemented in nano-seconds
   * as we are using TAI time format, but in practice platforms such as Windows
   * do not properly store nano-seconds, so the best you can probably rely on is milliseconds.
   */
  itkSetMacro(TimeStampTolerance, unsigned long int);
  itkGetMacro(TimeStampTolerance, unsigned long int);

  /**
   * \brief Sets the data storage, as each data source can put items into the storage.
   */
  itkSetObjectMacro(DataStorage, mitk::DataStorage);
  itkGetConstMacro(DataStorage, mitk::DataStorage*);

  /**
   * \brief Sets the file name prefix, for where to save data, as each
   * source can decide where and how to dump data to disk.
   */
  itkSetMacro(SavePrefix, std::string);
  itkGetMacro(SavePrefix, std::string);

  /**
   * \brief Returns true if we are saving messages and false otherwise.
   */
  itkGetMacro(SavingMessages, bool);
  virtual void SetSavingMessages(bool isSaving);

  /**
   * \brief If set to true, the data is saved in a background thread, and if false it is saved synchronously immediately.
   */
  itkSetMacro(SaveInBackground, bool);
  itkGetMacro(SaveInBackground, bool);

  /**
   * \brief If set to true, we save when the data is received, and if false, only when we
   * update the GUI, which may be at a different refresh rate to the incoming data.
   */
  itkSetMacro(SaveOnReceipt, bool);
  itkGetMacro(SaveOnReceipt, bool);

  /**
   * \brief FrameRate is calculated internally, and can be retrieved here in frames per second.
   */
  itkGetMacro(FrameRate, float);

  /**
   * \brief Get the time stamp of the most recently requested time-point.
   */
  igtlUint64 GetRequestedTimeStamp() const { return GetTimeInNanoSeconds(m_RequestedTimeStamp); }

  /**
   * \brief This is calculated internally, and represents the time-stamp of the "current" message,
   * which may be before or after that returned by GetRequestedTimeStamp()..
   */
  igtlUint64 GetActualTimeStamp() const { return GetTimeInNanoSeconds(m_ActualTimeStamp); }

  /**
   * \brief Returns the current data item that corresponds to the GetActualTimeStamp(),
   * with no searching to find a new one, and no updating of any buffer pointers or timestamps.
   */
  itkGetMacro(ActualData, mitk::IGIDataType::Pointer);

  /**
   * \brief Tools can have an optional Initialize function to perform any setup after construction,
   * with this class providing a default, do-nothing implementation.
   */
  virtual void Initialize() {};

  /**
   * \brief Derived classes can update the frame rate, as they receive data,
   * and the units should be in frames per second.
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
   * \brief Method to iterate through the buffer, and call DoSaveData() on each
   * item that needs saving, returning the number of saved messages.
   */
  unsigned long int SaveBuffer();

  /**
   * \brief Returns the number of items in the buffer to aid testing.
   */
  unsigned long int GetBufferSize() const;

  /**
   * \brief Returns the first time stamp, or 0 if the buffer is empty.
   */
  igtlUint64 GetFirstTimeStamp() const;

  /**
   * \brief Returns the last time stamp, or 0 if the buffer is empty.
   */
  igtlUint64 GetLastTimeStamp() const;

  /**
   * \brief Add data to the buffer, derived classes can decide to reject the data,
   * so returns true if added and false otherwise.
   */
  bool AddData(mitk::IGIDataType* data);

  /**
   * \brief Processes the data for a given timestamp, returning true if
   * the current data is processed successfully and within time tolerance.
   */
  bool ProcessData(igtlUint64 requestedTimeStamp);

  /**
   * \brief Returns true if the current data frame is within time tolerances and false otherwise.
   */
  bool IsCurrentWithinTimeTolerance() const;

  /**
   * \brief Returns the difference between the currentTimeStamp, and the GetActualTimeStamp(), and converts to seconds.
   */
  double GetCurrentTimeLag();
  /**
   * \brief Get the subtool list
   */
  std::list<std::string>  GetSubToolList ( ) ;

protected:

  IGIDataSource(); // Purposefully hidden.
  virtual ~IGIDataSource(); // Purposefully hidden.

  IGIDataSource(const IGIDataSource&); // Purposefully not implemented.
  IGIDataSource& operator=(const IGIDataSource&); // Purposefully not implemented.

  /**
   * \brief Derived classes should implement this to decide if they can handle a certain type of data,
   * returning true if it can be handled, and false otherwise.
   */
  virtual bool CanHandleData(mitk::IGIDataType* data) const = 0;

  /**
   * \brief Derived classes implement this to provide some kind of update based on the given data,
   * which must be NOT NULL, and should return true for successful and false otherwise.
   *
   * This method has default, do-nothing implementation to make unit testing of the internal buffers
   * and of the time-stamping mechanism easier, as we can create a test class derived from this one.
   */
  virtual bool Update(mitk::IGIDataType* data) { return true; }

  /**
   * \brief Derived classes may implement this to provide a mechanism to save data.
   * \param data the data type, which is guaranteed to be valid for the subclass when this method is called (i.e. CanHandle has already returned true).
   * \param outputFileName this will be written to, and contain the filename that the derived class actually attempted writing to.
   * \return true if the file was saved, and false otherwise.
   */
  virtual bool SaveData(mitk::IGIDataType* data, std::string& outputFileName) { return true; }

  /**
   * \brief Will iterate through the internal buffer, returning the
   * closest message to the requested time stamp, and sets the
   * ActualTimeStamp accordingly, or else return NULL if it can't be found.
   */
  virtual mitk::IGIDataType* RequestData(igtlUint64 requestedTimeStamp);

  /**
   * \brief Function to set the list of sub tools
   */
  void SetToolStringList ( std::list<std::string> );


private:

  /**
   * \brief Private method that takes the input data, asks derived classes to save it
   * and then stamps it the data object with the save status and filename.
   */
  bool DoSaveData(mitk::IGIDataType* data);

  itk::FastMutexLock::Pointer                     m_Mutex;
  mitk::DataStorage                              *m_DataStorage;
  int                                             m_Identifier;
  float                                           m_FrameRate;
  unsigned long int                               m_CurrentFrameId;
  std::string                                     m_Name;
  std::string                                     m_Type;
  std::string                                     m_Status;
  std::string                                     m_Description;
  bool                                            m_SavingMessages;
  bool                                            m_SaveOnReceipt;
  bool                                            m_SaveInBackground;
  std::string                                     m_SavePrefix;
  std::list<mitk::IGIDataType::Pointer>           m_Buffer;
  std::list<mitk::IGIDataType::Pointer>::iterator m_BufferIterator;
  std::list<mitk::IGIDataType::Pointer>::iterator m_FrameRateBufferIterator;
  igtl::TimeStamp::Pointer                        m_RequestedTimeStamp;
  igtl::TimeStamp::Pointer                        m_ActualTimeStamp;
  unsigned long int                               m_TimeStampTolerance;
  mitk::IGIDataType*                              m_ActualData;
  int                                             m_NumberOfTools;
  std::list<std::string>                          m_SubTools;
  std::list<std::string>::iterator                m_SubToolsIterator;

}; // end class

} // end namespace

#endif

