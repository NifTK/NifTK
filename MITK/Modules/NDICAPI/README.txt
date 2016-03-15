ndicapi was copied from PLUS svn revision 4564.

As we understand it, PLUS (https://www.assembla.com/spaces/plus/wiki)
is provided under BSD-style license, and in turn PLUS has used
the Atami code, also provided under a BSD-style license. 

This NifTK module contains niftkNDICAPITracker,
which was created by taking vtkNDITracker from PLUS
and trying to make the minimum number of changes to
get it to work. As such niftkNDICAPITracker is a
derived work of vtkNDITracker, which itself is a
wrapper around Atami's ndicapi.

We do not want to try and duplicate all the functionality
in PLUS. PLUS interfaces with many devices. Go
and visit the PLUS website. If PLUS integrates with your
device of choice, then you could use PLUS to grab your
data and stream it to NifTK via NiftyLink (using OpenIGTLink).

That said, in this specific case, a direct connection
to NDI Aurora and NDI Spectra would be very useful indeed.
So, for JUST this use-case, we are utilising this code.
