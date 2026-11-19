`roman_siaf_cgi.xml` holds the CGI aperture parameters of the Roman SIAF that `roman_pointing.getSIAFAperture` reads: for each CGI aperture, its `AperName`, `V2Ref`, `V3Ref` and `V3IdlYAngle`.
The current values are from the in-flight SIAF release `roman_siaf_inflight_20261021.xml`.
To update them from a new release, run `tools/extract_siaf_cgi.py` (see its docstring) with the same output file name, and record the release name here.
