`roman_siaf_inflight_20261021_CGI.xml` contains the release header and CGI `SiafEntry` blocks of the Roman SIAF, read by `roman_pointing.getSIAFAperture`.
It was extracted from the in-flight SIAF release `roman_siaf_inflight_20261021.xml`.
Regenerate it from a new release with `tools/extract_siaf_cgi.py` (see its docstring), then update `SIAF_FILE` in `roman_pointing.py`.
