"""
Copy the CGI entries of a Roman SIAF release to a small XML file for roman_pointing.getSIAFAperture.

The release header and change-history comments, which precede the root element, are kept; the CGI SiafEntry
blocks are copied byte-for-byte.

Usage:
  python tools/extract_siaf_cgi.py roman_siaf_inflight_20261021.xml \\
      src/roman_pointing/data/roman_siaf_inflight_20261021_CGI.xml

"""

import os
import re
import sys

src, out = sys.argv[1:]
text = open(src, encoding="utf-8").read()
head = text[: text.index("<SiafEntry>")]
cgi = re.findall(r"<SiafEntry>\s*<InstrName>CGI</InstrName>.*?</SiafEntry>", text, flags=re.S)
with open(out, "w", encoding="utf-8") as f:
    f.write(head.replace("<SIAF>", f"<!--CGI entries extracted from {os.path.basename(src)}-->\n<SIAF>")
            + "\n\t".join(cgi) + "\n</SIAF>\n")
