"""
Write the CGI aperture parameters of a Roman SIAF release to the small XML file read by roman_pointing.getSIAFAperture.

Only the parameters roman_pointing uses are kept: for each CGI SiafEntry, its AperName and the V2Ref, V3Ref and
V3IdlYAngle values, copied as text from the release. Record the release name in src/roman_pointing/siaf/README.md.

Usage:
  python tools/extract_siaf_cgi.py roman_siaf_inflight_20261021.xml src/roman_pointing/siaf/roman_siaf_cgi.xml

"""

import os
import sys
import xml.etree.ElementTree as ET

src, out = sys.argv[1:]
lines = ["<?xml version='1.0' encoding='UTF-8'?>", "<SIAF>"]
for entry in ET.parse(src).getroot().iter("SiafEntry"):
    if entry.findtext("InstrName") == "CGI":
        lines.append("\t<SiafEntry>")
        for key in ("AperName", "V2Ref", "V3Ref", "V3IdlYAngle"):
            lines.append(f"\t\t<{key}>{entry.findtext(key)}</{key}>")
        lines.append("\t</SiafEntry>")
lines.append("</SIAF>")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {out} from {os.path.basename(src)}; record the release name in src/roman_pointing/siaf/README.md")
