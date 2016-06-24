/****************************************************************************
** Copyright (c) 2000-2003 Wayne Roth
** Copyright (c) 2004-2007 Stefan Sander
** Copyright (c) 2007 Michal Policht
** Copyright (c) 2008 Brandon Fosdick
** Copyright (c) 2009-2010 Liam Staskawicz
** Copyright (c) 2011 Debao Zhang
** All right reserved.
** Web: http://code.google.com/p/qextserialport/
**
** Permission is hereby granted, free of charge, to any person obtaining
** a copy of this software and associated documentation files (the
** "Software"), to deal in the Software without restriction, including
** without limitation the rights to use, copy, modify, merge, publish,
** distribute, sublicense, and/or sell copies of the Software, and to
** permit persons to whom the Software is furnished to do so, subject to
** the following conditions:
**
** The above copyright notice and this permission notice shall be
** included in all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
** EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
** MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
** NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
** LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
** OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
** WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**
****************************************************************************/

#if defined(_WIN32) || defined(_WIN64)

#include "qextserialenumerator.h"
#include "qextserialenumerator_p.h"
#include <QtCore/QDebug>
#include <QtCore/QMetaType>
#include <QtCore/QRegExp>
#include <objbase.h>
#include <initguid.h>
#include <setupapi.h>
#include <dbt.h>
//#include "qextserialport.h"

#ifdef HAS_QWIDGET
#include <QWidget>
class QextSerialRegistrationWidget : public QWidget
{
public:
    QextSerialRegistrationWidget(QextSerialEnumeratorPrivate* qese) {
        this->qese = qese;
    }
    ~QextSerialRegistrationWidget() {}

protected:
    bool winEvent( MSG* message, long* result ) {
        if ( message->message == WM_DEVICECHANGE ) {
            qese->onDeviceChanged(message->wParam, message->lParam );
            *result = 1;
            return true;
        }
        return false;
    }
private:
    QextSerialEnumeratorPrivate* qese;
};

#endif // HAS_QWIDGET

void QextSerialEnumeratorPrivate::platformSpecificInit()
{
#ifdef HAS_QWIDGET
    notificationWidget = 0;
#endif // HAS_QWIDGET
}

/*!
  default
*/
void QextSerialEnumeratorPrivate::platformSpecificDestruct()
{
#ifdef HAS_QWIDGET
    if( notificationWidget )
        delete notificationWidget;
#endif
}

// ### This url has broken, anyone can fix it?
// see http://msdn.microsoft.com/en-us/library/ms791134.aspx for list of GUID classes
#ifndef GUID_DEVCLASS_PORTS
    DEFINE_GUID(GUID_DEVCLASS_PORTS, 0x4D36E978, 0xE325, 0x11CE, 0xBF, 0xC1, 0x08, 0x00, 0x2B, 0xE1, 0x03, 0x18 );
#endif

/* Gordon Schumacher's macros for TCHAR -> QString conversions and vice versa */
#ifdef UNICODE
    #define QStringToTCHAR(x)     (wchar_t*) x.utf16()
    #define PQStringToTCHAR(x)    (wchar_t*) x->utf16()
    #define TCHARToQString(x)     QString::fromUtf16((ushort*)(x))
    #define TCHARToQStringN(x,y)  QString::fromUtf16((ushort*)(x),(y))
#else
    #define QStringToTCHAR(x)     x.local8Bit().constData()
    #define PQStringToTCHAR(x)    x->local8Bit().constData()
    #define TCHARToQString(x)     QString::fromLocal8Bit((char*)(x))
    #define TCHARToQStringN(x,y)  QString::fromLocal8Bit((char*)(x),(y))
#endif /*UNICODE*/

/*!
    \internal
    Get value of specified property from the registry.
        \a key handle to an open key.
        \a property property name.

        return property value.
*/
static QString getRegKeyValue(HKEY key, LPCTSTR property)
{
    DWORD size = 0;
    DWORD type;
    ::RegQueryValueEx(key, property, NULL, NULL, NULL, & size);
    BYTE* buff = new BYTE[size];
    QString result;
    if(::RegQueryValueEx(key, property, NULL, &type, buff, & size) == ERROR_SUCCESS )
        result = TCHARToQString(buff);
    ::RegCloseKey(key);
    delete [] buff;
    return result;
}

/*!
     \internal
     Get specific property from registry.
     \a devInfo pointer to the device information set that contains the interface
        and its underlying device. Returned by SetupDiGetClassDevs() function.
     \a devData pointer to an SP_DEVINFO_DATA structure that defines the device instance.
        this is returned by SetupDiGetDeviceInterfaceDetail() function.
     \a property registry property. One of defined SPDRP_* constants.

     return property string.
 */
static QString getDeviceProperty(HDEVINFO devInfo, PSP_DEVINFO_DATA devData, DWORD property)
{
    DWORD buffSize = 0;
    ::SetupDiGetDeviceRegistryProperty(devInfo, devData, property, NULL, NULL, 0, & buffSize);
    BYTE* buff = new BYTE[buffSize];
    ::SetupDiGetDeviceRegistryProperty(devInfo, devData, property, NULL, buff, buffSize, NULL);
    QString result = TCHARToQString(buff);
    delete [] buff;
    return result;
}

/*!
     \internal
*/
static bool getDeviceDetailsWin( QextPortInfo* portInfo, HDEVINFO devInfo, PSP_DEVINFO_DATA devData
                                 , WPARAM wParam = DBT_DEVICEARRIVAL)
{
    portInfo->friendName = getDeviceProperty(devInfo, devData, SPDRP_FRIENDLYNAME);
    if( wParam == DBT_DEVICEARRIVAL)
        portInfo->physName = getDeviceProperty(devInfo, devData, SPDRP_PHYSICAL_DEVICE_OBJECT_NAME);
    portInfo->enumName = getDeviceProperty(devInfo, devData, SPDRP_ENUMERATOR_NAME);
    QString hardwareIDs = getDeviceProperty(devInfo, devData, SPDRP_HARDWAREID);
    HKEY devKey = ::SetupDiOpenDevRegKey(devInfo, devData, DICS_FLAG_GLOBAL, 0, DIREG_DEV, KEY_QUERY_VALUE);
    portInfo->portName = getRegKeyValue(devKey, TEXT("PortName"));
    QRegExp idRx(QLatin1String("VID_(\\w+)&PID_(\\w+)"));
    if(hardwareIDs.toUpper().contains(idRx)) {
        bool dummy;
        portInfo->vendorID = idRx.cap(1).toInt(&dummy, 16);
        portInfo->productID = idRx.cap(2).toInt(&dummy, 16);
        //qDebug() << "got vid:" << vid << "pid:" << pid;
    }
    return true;
}

/*!
     \internal
*/
static void enumerateDevicesWin( const GUID & guid, QList<QextPortInfo>* infoList )
{
    HDEVINFO devInfo;
    if( (devInfo = ::SetupDiGetClassDevs(&guid, NULL, NULL, DIGCF_PRESENT)) != INVALID_HANDLE_VALUE) {
        SP_DEVINFO_DATA devInfoData;
        devInfoData.cbSize = sizeof(SP_DEVINFO_DATA);
        for(int i = 0; ::SetupDiEnumDeviceInfo(devInfo, i, &devInfoData); i++) {
            QextPortInfo info;
            info.productID = info.vendorID = 0;
            getDeviceDetailsWin( &info, devInfo, &devInfoData );
            infoList->append(info);
        }
        ::SetupDiDestroyDeviceInfoList(devInfo);
    }
}


static bool lessThan(const QextPortInfo &s1, const QextPortInfo &s2)
{
    if (s1.portName.startsWith(QLatin1String("COM"))
            && s2.portName.startsWith(QLatin1String("COM"))) {
        return s1.portName.mid(3).toInt()<s2.portName.mid(3).toInt();
    }
    return s1.portName < s2.portName;
}


/*!
    Get list of ports.

    return list of ports currently available in the system.
*/
QList<QextPortInfo> QextSerialEnumeratorPrivate::getPorts_sys()
{
    QList<QextPortInfo> ports;
    enumerateDevicesWin(GUID_DEVCLASS_PORTS, &ports);
    qSort(ports.begin(), ports.end(), lessThan);
    return ports;
}


/*
    Enable event-driven notifications of board discovery/removal.
*/
bool QextSerialEnumeratorPrivate::setUpNotifications_sys(bool setup)
{
#ifndef HAS_QWIDGET
    Q_UNUSED(setup)
    QESP_WARNING("QextSerialEnumerator: GUI not enabled - can't register for device notifications.");
    return false;
#else
    Q_Q(QextSerialEnumerator);
    if(setup && notificationWidget) //already setup
        return true;
    notificationWidget = new QextSerialRegistrationWidget(this);

    DEV_BROADCAST_DEVICEINTERFACE dbh;
    ::ZeroMemory(&dbh, sizeof(dbh));
    dbh.dbcc_size = sizeof(dbh);
    dbh.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE;
    ::CopyMemory(&dbh.dbcc_classguid, &GUID_DEVCLASS_PORTS, sizeof(GUID));
    if(::RegisterDeviceNotification((HWND)notificationWidget->winId(), &dbh, DEVICE_NOTIFY_WINDOW_HANDLE ) == NULL) {
        QESP_WARNING() << "RegisterDeviceNotification failed:" << GetLastError();
        return false;
    }
    // setting up notifications doesn't tell us about devices already connected
    // so get those manually
    foreach(QextPortInfo port, getPorts_sys())
      Q_EMIT q->deviceDiscovered(port);
    return true;
#endif
}

LRESULT QextSerialEnumeratorPrivate::onDeviceChanged( WPARAM wParam, LPARAM lParam )
{
    if (DBT_DEVICEARRIVAL == wParam || DBT_DEVICEREMOVECOMPLETE == wParam ) {
        PDEV_BROADCAST_HDR pHdr = (PDEV_BROADCAST_HDR)lParam;
        if(pHdr->dbch_devicetype == DBT_DEVTYP_DEVICEINTERFACE ) {
            PDEV_BROADCAST_DEVICEINTERFACE pDevInf = (PDEV_BROADCAST_DEVICEINTERFACE)pHdr;
             // delimiters are different across APIs...change to backslash.  ugh.
            QString deviceID = TCHARToQString(pDevInf->dbcc_name).toUpper().replace(QLatin1String("#"), QLatin1String("\\"));

            matchAndDispatchChangedDevice(deviceID, GUID_DEVCLASS_PORTS, wParam);
        }
    }
    return 0;
}

bool QextSerialEnumeratorPrivate::matchAndDispatchChangedDevice(const QString & deviceID, const GUID & guid, WPARAM wParam)
{
    Q_Q(QextSerialEnumerator);
    bool rv = false;
    DWORD dwFlag = (DBT_DEVICEARRIVAL == wParam) ? DIGCF_PRESENT : DIGCF_ALLCLASSES;
    HDEVINFO devInfo;
    if( (devInfo = SetupDiGetClassDevs(&guid,NULL,NULL,dwFlag)) != INVALID_HANDLE_VALUE ) {
        SP_DEVINFO_DATA spDevInfoData;
        spDevInfoData.cbSize = sizeof(SP_DEVINFO_DATA);
        for(int i=0; SetupDiEnumDeviceInfo(devInfo, i, &spDevInfoData); i++) {
            DWORD nSize=0;
            TCHAR buf[MAX_PATH];
            if ( SetupDiGetDeviceInstanceId(devInfo, &spDevInfoData, buf, MAX_PATH, &nSize) &&
                    deviceID.contains(TCHARToQString(buf))) { // we found a match
                rv = true;
                QextPortInfo info;
                info.productID = info.vendorID = 0;
                getDeviceDetailsWin( &info, devInfo, &spDevInfoData, wParam );
                if( wParam == DBT_DEVICEARRIVAL )
                    Q_EMIT q->deviceDiscovered(info);
                else if( wParam == DBT_DEVICEREMOVECOMPLETE )
                    Q_EMIT q->deviceRemoved(info);
                break;
            }
        }
        SetupDiDestroyDeviceInfoList(devInfo);
    }
    return rv;
}

#endif  //Q_OS_WIN

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include "qextserialenumerator.h"
#include "qextserialenumerator_p.h"
#include <QtCore/QDebug>
#include <QtCore/QStringList>
#include <QtCore/QDir>

void QextSerialEnumeratorPrivate::platformSpecificInit()
{
}

void QextSerialEnumeratorPrivate::platformSpecificDestruct()
{
}

QList<QextPortInfo> QextSerialEnumeratorPrivate::getPorts_sys()
{
    QList<QextPortInfo> infoList;
#ifdef Q_OS_LINUX
    QStringList portNamePrefixes, portNameList;
    portNamePrefixes << QLatin1String("ttyS*"); // list normal serial ports first

    QDir dir(QLatin1String("/dev"));
    portNameList = dir.entryList(portNamePrefixes, (QDir::System | QDir::Files), QDir::Name);

    // remove the values which are not serial ports for e.g.  /dev/ttysa
    for (int i = 0; i < portNameList.size(); i++) {
        bool ok;
        QString current = portNameList.at(i);
        // remove the ttyS part, and check, if the other part is a number
        current.remove(0,4).toInt(&ok, 10);
        if (!ok) {
            portNameList.removeAt(i);
            i--;
        }
    }

    // get the non standard serial ports names
    // (USB-serial, bluetooth-serial, 18F PICs, and so on)
    // if you know an other name prefix for serial ports please let us know
    portNamePrefixes.clear();
    portNamePrefixes << QLatin1String("ttyACM*") << QLatin1String("ttyUSB*") << QLatin1String("rfcomm*");
    portNameList += dir.entryList(portNamePrefixes, (QDir::System | QDir::Files), QDir::Name);

    foreach (QString str , portNameList) {
        QextPortInfo inf;
        inf.physName = QLatin1String("/dev/")+str;
        inf.portName = str;

        if (str.contains(QLatin1String("ttyS"))) {
            inf.friendName = QLatin1String("Serial port ")+str.remove(0, 4);
        }
        else if (str.contains(QLatin1String("ttyUSB"))) {
            inf.friendName = QLatin1String("USB-serial adapter ")+str.remove(0, 6);
        }
        else if (str.contains(QLatin1String("rfcomm"))) {
            inf.friendName = QLatin1String("Bluetooth-serial adapter ")+str.remove(0, 6);
        }
        inf.enumName = QLatin1String("/dev"); // is there a more helpful name for this?
        infoList.append(inf);
    }
#endif
    return infoList;
}

bool QextSerialEnumeratorPrivate::setUpNotifications_sys(bool setup)
{
    Q_UNUSED(setup)
    return false;
}

#endif //Q_OS_UNIX

#ifdef __APPLE__


#include "qextserialenumerator.h"
#include "qextserialenumerator_p.h"
#include <QtCore/QDebug>
#include <IOKit/serial/IOSerialKeys.h>
#include <IOKit/IOKitKeys.h>
#include <CoreFoundation/CFNumber.h>
#include <sys/param.h>

void QextSerialEnumeratorPrivate::platformSpecificInit()
{
}

void QextSerialEnumeratorPrivate::platformSpecificDestruct()
{
    IONotificationPortDestroy( notificationPortRef );
}

// static
QList<QextPortInfo> QextSerialEnumeratorPrivate::getPorts_sys()
{
    QList<QextPortInfo> infoList;
    io_iterator_t serialPortIterator = 0;
    kern_return_t kernResult = KERN_FAILURE;
    CFMutableDictionaryRef matchingDictionary;

    // first try to get any serialbsd devices, then try any USBCDC devices
    if( !(matchingDictionary = IOServiceMatching(kIOSerialBSDServiceValue) ) ) {
        QESP_WARNING("IOServiceMatching returned a NULL dictionary.");
        return infoList;
    }
    CFDictionaryAddValue(matchingDictionary, CFSTR(kIOSerialBSDTypeKey), CFSTR(kIOSerialBSDAllTypes));

    // then create the iterator with all the matching devices
    if( IOServiceGetMatchingServices(kIOMasterPortDefault, matchingDictionary, &serialPortIterator) != KERN_SUCCESS ) {
        qCritical() << "IOServiceGetMatchingServices failed, returned" << kernResult;
        return infoList;
    }
    iterateServicesOSX(serialPortIterator, infoList);
    IOObjectRelease(serialPortIterator);
    serialPortIterator = 0;

    if( !(matchingDictionary = IOServiceNameMatching("AppleUSBCDC")) ) {
        QESP_WARNING("IOServiceNameMatching returned a NULL dictionary.");
        return infoList;
    }

    if( IOServiceGetMatchingServices(kIOMasterPortDefault, matchingDictionary, &serialPortIterator) != KERN_SUCCESS ) {
        qCritical() << "IOServiceGetMatchingServices failed, returned" << kernResult;
        return infoList;
    }
    iterateServicesOSX(serialPortIterator, infoList);
    IOObjectRelease(serialPortIterator);

    return infoList;
}

void QextSerialEnumeratorPrivate::iterateServicesOSX(io_object_t service, QList<QextPortInfo> & infoList)
{
    // Iterate through all modems found.
    io_object_t usbService;
    while( ( usbService = IOIteratorNext(service) ) )
    {
        QextPortInfo info;
        info.vendorID = 0;
        info.productID = 0;
        getServiceDetailsOSX( usbService, &info );
        infoList.append(info);
    }
}

bool QextSerialEnumeratorPrivate::getServiceDetailsOSX( io_object_t service, QextPortInfo* portInfo )
{
    bool retval = true;
    CFTypeRef bsdPathAsCFString = NULL;
    CFTypeRef productNameAsCFString = NULL;
    CFTypeRef vendorIdAsCFNumber = NULL;
    CFTypeRef productIdAsCFNumber = NULL;
    // check the name of the modem's callout device
    bsdPathAsCFString = IORegistryEntryCreateCFProperty(service, CFSTR(kIOCalloutDeviceKey),
                                                        kCFAllocatorDefault, 0);

    // wander up the hierarchy until we find the level that can give us the
    // vendor/product IDs and the product name, if available
    io_registry_entry_t parent;
    kern_return_t kernResult = IORegistryEntryGetParentEntry(service, kIOServicePlane, &parent);
    while( kernResult == KERN_SUCCESS && !vendorIdAsCFNumber && !productIdAsCFNumber )
    {
        if(!productNameAsCFString)
            productNameAsCFString = IORegistryEntrySearchCFProperty(parent,
                                                                    kIOServicePlane,
                                                                    CFSTR("Product Name"),
                                                                    kCFAllocatorDefault, 0);
        vendorIdAsCFNumber = IORegistryEntrySearchCFProperty(parent,
                                                             kIOServicePlane,
                                                             CFSTR(kUSBVendorID),
                                                             kCFAllocatorDefault, 0);
        productIdAsCFNumber = IORegistryEntrySearchCFProperty(parent,
                                                              kIOServicePlane,
                                                              CFSTR(kUSBProductID),
                                                              kCFAllocatorDefault, 0);
        io_registry_entry_t oldparent = parent;
        kernResult = IORegistryEntryGetParentEntry(parent, kIOServicePlane, &parent);
        IOObjectRelease(oldparent);
    }

    io_string_t ioPathName;
    IORegistryEntryGetPath( service, kIOServicePlane, ioPathName );
    portInfo->physName = ioPathName;

    if( bsdPathAsCFString )
    {
        char path[MAXPATHLEN];
        if( CFStringGetCString((CFStringRef)bsdPathAsCFString, path,
                               PATH_MAX, kCFStringEncodingUTF8) )
            portInfo->portName = path;
        CFRelease(bsdPathAsCFString);
    }

    if(productNameAsCFString)
    {
        char productName[MAXPATHLEN];
        if( CFStringGetCString((CFStringRef)productNameAsCFString, productName,
                               PATH_MAX, kCFStringEncodingUTF8) )
            portInfo->friendName = productName;
        CFRelease(productNameAsCFString);
    }

    if(vendorIdAsCFNumber)
    {
        SInt32 vID;
        if(CFNumberGetValue((CFNumberRef)vendorIdAsCFNumber, kCFNumberSInt32Type, &vID))
            portInfo->vendorID = vID;
        CFRelease(vendorIdAsCFNumber);
    }

    if(productIdAsCFNumber)
    {
        SInt32 pID;
        if(CFNumberGetValue((CFNumberRef)productIdAsCFNumber, kCFNumberSInt32Type, &pID))
            portInfo->productID = pID;
        CFRelease(productIdAsCFNumber);
    }
    IOObjectRelease(service);
    return retval;
}

// IOKit callbacks registered via setupNotifications()
void deviceDiscoveredCallbackOSX( void *ctxt, io_iterator_t serialPortIterator )
{
    QextSerialEnumeratorPrivate* d = (QextSerialEnumeratorPrivate*)ctxt;
    io_object_t serialService;
    while ((serialService = IOIteratorNext(serialPortIterator)))
        d->onDeviceDiscoveredOSX(serialService);
}

void deviceTerminatedCallbackOSX( void *ctxt, io_iterator_t serialPortIterator )
{
    QextSerialEnumeratorPrivate* d = (QextSerialEnumeratorPrivate*)ctxt;
    io_object_t serialService;
    while ((serialService = IOIteratorNext(serialPortIterator)))
        d->onDeviceTerminatedOSX(serialService);
}

/*
  A device has been discovered via IOKit.
  Create a QextPortInfo if possible, and emit the signal indicating that we've found it.
*/
void QextSerialEnumeratorPrivate::onDeviceDiscoveredOSX( io_object_t service )
{
    Q_Q(QextSerialEnumerator);
    QextPortInfo info;
    info.vendorID = 0;
    info.productID = 0;
    if( getServiceDetailsOSX( service, &info ) )
        Q_EMIT q->deviceDiscovered( info );
}

/*
  Notification via IOKit that a device has been removed.
  Create a QextPortInfo if possible, and emit the signal indicating that it's gone.
*/
void QextSerialEnumeratorPrivate::onDeviceTerminatedOSX( io_object_t service )
{
    Q_Q(QextSerialEnumerator);
    QextPortInfo info;
    info.vendorID = 0;
    info.productID = 0;
    if( getServiceDetailsOSX( service, &info ) )
        Q_EMIT q->deviceRemoved( info );
}

/*
  Create matching dictionaries for the devices we want to get notifications for,
  and add them to the current run loop.  Invoke the callbacks that will be responding
  to these notifications once to arm them, and discover any devices that
  are currently connected at the time notifications are setup.
*/
bool QextSerialEnumeratorPrivate::setUpNotifications_sys(bool setup)
{
    kern_return_t kernResult;
    mach_port_t masterPort;
    CFRunLoopSourceRef notificationRunLoopSource;
    CFMutableDictionaryRef classesToMatch;
    CFMutableDictionaryRef cdcClassesToMatch;
    io_iterator_t portIterator;

    kernResult = IOMasterPort(MACH_PORT_NULL, &masterPort);
    if (KERN_SUCCESS != kernResult) {
        qDebug() << "IOMasterPort returned:" << kernResult;
        return false;
    }

    classesToMatch = IOServiceMatching(kIOSerialBSDServiceValue);
    if (classesToMatch == NULL)
        qDebug("IOServiceMatching returned a NULL dictionary.");
    else
        CFDictionarySetValue(classesToMatch, CFSTR(kIOSerialBSDTypeKey), CFSTR(kIOSerialBSDAllTypes));

    if( !(cdcClassesToMatch = IOServiceNameMatching("AppleUSBCDC") ) ) {
        QESP_WARNING("couldn't create cdc matching dict");
        return false;
    }

    // Retain an additional reference since each call to IOServiceAddMatchingNotification consumes one.
    classesToMatch = (CFMutableDictionaryRef) CFRetain(classesToMatch);
    cdcClassesToMatch = (CFMutableDictionaryRef) CFRetain(cdcClassesToMatch);

    notificationPortRef = IONotificationPortCreate(masterPort);
    if(notificationPortRef == NULL) {
        qDebug("IONotificationPortCreate return a NULL IONotificationPortRef.");
        return false;
    }

    notificationRunLoopSource = IONotificationPortGetRunLoopSource(notificationPortRef);
    if (notificationRunLoopSource == NULL) {
        qDebug("IONotificationPortGetRunLoopSource returned NULL CFRunLoopSourceRef.");
        return false;
    }

    CFRunLoopAddSource(CFRunLoopGetCurrent(), notificationRunLoopSource, kCFRunLoopDefaultMode);

    kernResult = IOServiceAddMatchingNotification(notificationPortRef, kIOMatchedNotification, classesToMatch,
                                                  deviceDiscoveredCallbackOSX, this, &portIterator);
    if (kernResult != KERN_SUCCESS) {
        qDebug() << "IOServiceAddMatchingNotification return:" << kernResult;
        return false;
    }

    // arm the callback, and grab any devices that are already connected
    deviceDiscoveredCallbackOSX( this, portIterator );

    kernResult = IOServiceAddMatchingNotification(notificationPortRef, kIOMatchedNotification, cdcClassesToMatch,
                                                  deviceDiscoveredCallbackOSX, this, &portIterator);
    if (kernResult != KERN_SUCCESS) {
        qDebug() << "IOServiceAddMatchingNotification return:" << kernResult;
        return false;
    }

    // arm the callback, and grab any devices that are already connected
    deviceDiscoveredCallbackOSX( this, portIterator );

    kernResult = IOServiceAddMatchingNotification(notificationPortRef, kIOTerminatedNotification, classesToMatch,
                                                  deviceTerminatedCallbackOSX, this, &portIterator);
    if (kernResult != KERN_SUCCESS) {
        qDebug() << "IOServiceAddMatchingNotification return:" << kernResult;
        return false;
    }

    // arm the callback, and clear any devices that are terminated
    deviceTerminatedCallbackOSX( this, portIterator );

    kernResult = IOServiceAddMatchingNotification(notificationPortRef, kIOTerminatedNotification, cdcClassesToMatch,
                                                  deviceTerminatedCallbackOSX, this, &portIterator);
    if (kernResult != KERN_SUCCESS) {
        qDebug() << "IOServiceAddMatchingNotification return:" << kernResult;
        return false;
    }

    // arm the callback, and clear any devices that are terminated
    deviceTerminatedCallbackOSX( this, portIterator );
    return true;
}
 #endif /*Q_OS_MAC*/
