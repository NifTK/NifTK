/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGITrackerSource_h
#define QmitkIGITrackerSource_h

#include "niftkIGIGuiExports.h"
#include "QmitkIGINiftyLinkDataSource.h"
#include <mitkDataNode.h>
#include <mitkPointSet.h>
#include <set>
#include <map>
#include <string>


/**
 * \class QmitkIGITrackerSource
 * \brief Base class for IGI Tracker Tools.
 */
class NIFTKIGIGUI_EXPORT QmitkIGITrackerSource : public QmitkIGINiftyLinkDataSource
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGITrackerSource, QmitkIGINiftyLinkDataSource);
  mitkNewMacro2Param(QmitkIGITrackerSource, mitk::DataStorage*, NiftyLinkSocketObject *);

  /**
   * \brief Defined in base class, so we check that the data is in fact a NiftyLinkMessageType containing tracking data.
   * \see mitk::IGIDataSource::CanHandleData()
   */
  virtual bool CanHandleData(mitk::IGIDataType* data) const;

  /**
   * \brief Defined in base class, this is the method where we do any update based on the available data.
   * \see mitk::IGIDataSource::Update()
   */
  virtual bool Update(mitk::IGIDataType* data);

  /**
   * \brief Returns the latest status message, for informational purposes only.
   * This method is not thread safe.
   */
  QString GetStatusMessage() const { return m_StatusMessage; }

  // overridden from IGIDataSource
  virtual bool ProbeRecordedData(const std::string& path, igtlUint64* firstTimeStampInStore, igtlUint64* lastTimeStampInStore);
  virtual void StartPlayback(const std::string& path, igtlUint64 firstTimeStamp, igtlUint64 lastTimeStamp);
  virtual void StopPlayback();
  virtual void PlaybackData(igtlUint64 requestedTimeStamp);

protected:

  QmitkIGITrackerSource(mitk::DataStorage* storage, NiftyLinkSocketObject* socket); // Purposefully hidden.
  virtual ~QmitkIGITrackerSource(); // Purposefully hidden.

  QmitkIGITrackerSource(const QmitkIGITrackerSource&); // Purposefully not implemented.
  QmitkIGITrackerSource& operator=(const QmitkIGITrackerSource&); // Purposefully not implemented.

  /**
   * \brief \see IGIDataSource::SaveData();
   */
  virtual bool SaveData(mitk::IGIDataType* data, std::string& outputFileName);

protected slots:

  /**
   * \brief Main message handler routine for this tool, called by the signal from the socket.
   */
  virtual void InterpretMessage(NiftyLinkMessage::Pointer msg);

private:

  std::map<std::string, std::set<igtlUint64> >    m_PlaybackIndex;
  std::string                                     m_PlaybackDirectoryName;
  QString                                         m_StatusMessage;
}; // end class

#endif
