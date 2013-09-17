/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkIGIDataSource_h
#define mitkIGIDataSource_h

#include "niftkIGIExports.h"
#include <NiftyLinkUtils.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkMessage.h>
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <itkFastMutexLock.h>
#include <list>
#include <set>
#include "mitkIGIDataType.h"
#include <itkNifTKMacro.h>

namespace mitk {

/**
 * \class IGIDataSource
 * \brief Base class for IGI Data Sources, where Data Sources can provide video data,
 * ultrasound data, tracker data etc.
 *
 * NOTE: All timestamps should be in UTC format. Also, take care NOT to expose a pointer to the
 * igtl::TimeStamp object. You should only ever expose a copy of this data, or an equivalent
 * representation of it, i.e. if you Set/Get the igtlUint64 values, then NO-ONE can modify the
 * timestamp and set the time to TAI for example.
 */
class NIFTKIGI_EXPORT IGIDataSource : public itk::Object
{

public:

  enum SourceTypeEnum {
    SOURCE_TYPE_UNKNOWN,
    SOURCE_TYPE_TRACKER,
    SOURCE_TYPE_IMAGER,
    SOURCE_TYPE_FRAME_GRABBER,
    SOURCE_TYPE_NVIDIA_SDI
  };

  mitkClassMacro(IGIDataSource, itk::Object);

  /**
   * \brief Sources can have an optional Initialize function to perform any setup after construction,
   * with this class providing a default, do-nothing implementation.
   */
  virtual void Initialize() {};

  /**
   * \brief Sets the data storage, as each data source can put items into the storage.
   * This method is not thread safe, so should only be called once at startup.
   */
  itkSetObjectMacro(DataStorage, mitk::DataStorage);
  itkGetConstMacro(DataStorage, mitk::DataStorage*);

  /**
   * \brief Sets the identifier, which is just a tag to uniquely identify the source by (i.e. item number in a list for example).
   */
  itkThreadSafeSetMacro(Identifier, int);
  itkThreadSafeGetConstMacro(Identifier, int);

  /**
   * \brief Sets the source type as determined by the front end. This is a system identifier, and should not be set by the user.
   */
  itkThreadSafeSetMacro(SourceType, SourceTypeEnum);
  itkThreadSafeGetConstMacro(SourceType, SourceTypeEnum);

  /**
   * \brief Sets a name, useful for display purposes.
   */
  itkThreadSafeSetMacro(Name, std::string);
  itkThreadSafeGetConstMacro(Name, std::string);

  /**
   * \brief Sets a type, useful for display purposes. This can be set by the user.
   */
  itkThreadSafeSetMacro(Type, std::string);
  itkThreadSafeGetConstMacro(Type, std::string);

  /**
   * \brief Sets a status message, useful for display purposes.
   */
  itkThreadSafeSetMacro(Status, std::string);
  itkThreadSafeGetConstMacro(Status, std::string);

  /**
   * \brief Sets a description, useful for display purposes.
   */
  itkThreadSafeSetMacro(Description, std::string);
  itkThreadSafeGetConstMacro(Description, std::string);

  /**
   * \brief Sets the time tolerance for checking data, implemented in nano-seconds, but
   * in practice platforms such as Windows do not properly store nano-seconds,
   * so the best you can probably rely on is milliseconds.
   */
  itkThreadSafeSetMacro(TimeStampTolerance, igtlUint64);
  itkThreadSafeGetConstMacro(TimeStampTolerance, igtlUint64);

  /**
   * \brief Sets the file name prefix, for where to save data, as each
   * source can decide where and how to dump data to disk.
   */
  itkThreadSafeSetMacro(SavePrefix, std::string);
  itkThreadSafeGetConstMacro(SavePrefix, std::string);

  /**
   * \brief Sets whether we are currently saving messages.
   */
  itkThreadSafeSetMacro(SavingMessages, bool);
  itkThreadSafeGetConstMacro(SavingMessages, bool);

  /**
   * \brief If set to true, the data is saved in a background thread, and if false it is saved synchronously.
   */
  itkThreadSafeSetMacro(SaveInBackground, bool);

  /**
   * \brief CAREFUL: This method, read only, can be overridden in derived classes,
   so for example, you could always return true,
   */
  virtual bool GetSaveInBackground() const { return m_SaveInBackground; }

  /**
   * \brief If set to true, we save when the data is received, and if false, only when we
   * update the GUI, which may be at a different refresh rate to the incoming data.
   */
  itkThreadSafeSetMacro(SaveOnReceipt, bool);
  itkThreadSafeGetConstMacro(SaveOnReceipt, bool);

  /**
   * \brief Sets a flag that if true, will not search the buffer for the data matching
   * a given timestamp, but will always pick the data at the end of the queue.
   */
  itkThreadSafeSetMacro(PickLatestData, bool);
  itkThreadSafeGetConstMacro(PickLatestData, bool);

  /**
   * Whether the data source will call Update() as part of its processing.
   * This effectively prevents a DataStorage update, kinda like freeze-frame.
   */
  itkThreadSafeSetMacro(ShouldCallUpdate, bool);
  itkThreadSafeGetConstMacro(ShouldCallUpdate, bool);

  itkThreadSafeSetMacro(IsPlayingBack, bool);
  itkThreadSafeGetConstMacro(IsPlayingBack, bool);

  /**
   * \brief FrameRate is calculated internally, and can be retrieved here in frames per second.
   */
  itkThreadSafeGetConstMacro(FrameRate, float);

  /**
   * \brief Returns the current data item that corresponds to the GetActualTimeStamp(),
   * with no searching to find a new one, and no updating of any buffer pointers or timestamps.
   */
  itkThreadSafeGetConstMacro(ActualData, mitk::IGIDataType::Pointer);

  /**
   * \brief Concatenates the SavePrefix, Name and Identifier to a Unique directory name.
   */
  std::string GetSaveDirectoryName();

  /**
   * \brief Recalculates the frame rate based on the number of items received and stored in the buffer.
   */
  virtual float UpdateFrameRate();

  /**
   * \brief Get the time stamp of the most recently requested time-point.
   */
  igtlUint64 GetRequestedTimeStamp() const;

  /**
   * \brief This is calculated internally, and represents the time-stamp of the "closest" message to the,
   * most recently requested time stamp, and hence may be before or after that returned by GetRequestedTimeStamp(),
   * depending on the available data.
   */
  igtlUint64 GetActualTimeStamp() const;

  /**
   * \brief Clears the internal buffer, which means completely destroying all the contents.
   */
  void ClearBuffer();

  /**
   * \brief Iterates through the buffer, clearing data, up to either the current time-stamp,
   * or if we are saving data, until we hit a piece of data that has not yet been saved,
   * as the save method may be called from another thread.
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
   * \brief Atomic method to do all the setup required while holding one mutex lock, to instigate recording.
   */
  virtual void StartRecording(const std::string& directoryPrefix, const bool& saveInBackground, const bool& saveOnReceipt);

  /**
   * \brief Stops recording, but does not reset the SaveInBackground flag or the SaveOnReceipt flag.
   */
  virtual void StopRecording();

  /**
   * Checks whether the previously recorded data is readable, and returns the time-range for it.
   * Default implementation returns false, i.e. is not capable of playback.
   */
  virtual bool ProbeRecordedData(const std::string& path, igtlUint64* firstTimeStampInStore, igtlUint64* lastTimeStampInStore);


  virtual void StartPlayback(const std::string& path, igtlUint64 firstTimeStamp, igtlUint64 lastTimeStamp);
  virtual void StopPlayback();
  // RequestData() would be a candidate to fold this into. but it's currently not suitable.
  virtual void PlaybackData(igtlUint64 requestedTimeStamp);

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
   * \brief Returns the difference between the nowTime and the most recent data from the source.
   */
  double GetCurrentTimeLag(const igtlUint64& nowTime );

  /**
   * \brief Returns true if the current data frame is within time tolerances and false otherwise.
   */
  bool IsCurrentWithinTimeTolerance() const;

  /**
   * \brief Returns the list of related sources.
   */
  std::list<std::string> GetRelatedSources();


protected:

  IGIDataSource(mitk::DataStorage* storage); // Purposefully hidden.
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
  mitk::IGIDataType* RequestData(igtlUint64 requestedTimeStamp);

  /**
   * \brief Derived classes request a node for a given name. If the node does not exist, it will
   * be created with some default properties.
   * \param name if supplied the node will be assigned that name, and if empty, the node
   * will be given the name this->GetName().
   * \param addToDataStorage if true, will be added to data storage, if false, the caller can determine when to do it.
   */
  mitk::DataNode::Pointer GetDataNode(const std::string& name=std::string(), const bool& addToDataStorage=true);

  /**
   * \brief Sets the list of related sources.
   */
  void SetRelatedSources(const std::list<std::string>& listOfSourceNames);

  /**
   * \brief Protected members, so derived classes can access it without using the threadsafe Getters.
   */
  std::string        m_SavePrefix;
  std::string        m_Description;
  igtlUint64         m_TimeStampTolerance;
  mitk::DataStorage* m_DataStorage;
  bool               m_ShouldCallUpdate;
  bool               m_IsPlayingBack;

private:

  /**
   * \brief Private method that takes the input data, asks derived classes to save it
   * and then stamps it the data object with the save status and filename.
   */
  bool DoSaveData(mitk::IGIDataType* data);

  /**
   * \brief Private method that checks the ActualData timestamp.
   *
   * Its a separate method, so it can be called as a private method from
   * within this class, and hence there is no locking of the class m_Mutex.
   */
  bool IsWithinTimeTolerance() const;

  itk::FastMutexLock::Pointer                     m_Mutex;
  std::set<mitk::DataNode::Pointer>               m_DataNodes;

  int                                             m_Identifier;
  SourceTypeEnum                                  m_SourceType;
  float                                           m_FrameRate;
  unsigned long int                               m_CurrentFrameId;
  std::string                                     m_Name;
  std::string                                     m_Type;
  std::string                                     m_Status;
  bool                                            m_SavingMessages;
  bool                                            m_SaveOnReceipt;
  bool                                            m_SaveInBackground;
  bool                                            m_PickLatestData;

  struct TimeStampComparator
  {
    bool operator()(const mitk::IGIDataType::Pointer& a, const mitk::IGIDataType::Pointer& b);
  };

  typedef std::set<mitk::IGIDataType::Pointer, TimeStampComparator>    BufferType;
  BufferType              m_Buffer;
  BufferType::iterator    m_BufferIterator;
  BufferType::iterator    m_FrameRateBufferIterator;

  igtl::TimeStamp::Pointer                        m_RequestedTimeStamp;
  igtl::TimeStamp::Pointer                        m_ActualTimeStamp;
  mitk::IGIDataType*                              m_ActualData;
  std::list<std::string>                          m_RelatedSources;
}; // end class

} // end namespace

#endif
