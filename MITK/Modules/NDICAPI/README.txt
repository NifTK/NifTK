ndicapi was copied from PLUS svn revision 4564.

As we understand it, PLUS (https://www.assembla.com/spaces/plus/wiki)
is provided under BSD-style license, and in turn PLUS has used
the Atami code, also provided under a BSD-style license. 

So, this NifTK module is a wrapper around ndicapi, which 
we can then use. We do not want to try and duplicate PLUS.
PLUS are doing very well indeed at integrating lots of 
devices into PLUS Server. So, you may want to use PLUS
to grab data, and stream to NifTK via OpenIGTLink.

However, in this specific case, a direct connection
to NDI Aurora and NDI Spectra would be very useful indeed.
So, for JUST this use-case, we are utilising this code.
