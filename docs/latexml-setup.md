# LaTeXML for the Superpod runner

Mark7 invokes `latexmlmath` through `scripts/sfc_def_structure.bb` for the S11
formula-to-structure conversion. LaTeXML is a separate Perl tool, installed
outside the Python project environment. Conda, libmamba, and libmambapy are not
runner dependencies and do not belong in its Python requirements.

Install LaTeXML from [upstream GitHub](https://github.com/brucemiller/LaTeXML)
or follow the [official installation instructions](https://math.nist.gov/~BMiller/LaTeXML/get.html).
Use those instructions for the Perl modules and native libraries appropriate
to the run host, including libxml2 and libxslt.

For a GitHub source build, first install the upstream prerequisites, then:

```bash
git clone https://github.com/brucemiller/LaTeXML.git
cd LaTeXML
perl Makefile.PL
make
make test
sudo make install
```

For a host without administrative access, follow upstream's nonstandard-prefix
installation instructions and configure the executable and Perl module search
paths for that prefix. Ensure `latexmlmath` is on `PATH` inside the actual job.

The official instructions also support `sudo apt-get install latexml` on
Debian/Ubuntu. `scripts/linode-postsetup-deps.sh` uses that route if LaTeXML is
missing and root or passwordless sudo is available. Otherwise it stops with
upstream installation links; it does not install through Conda.

From the `futon6` checkout, verify the installation:

```bash
latexmlmath --VERSION
scripts/linode-postsetup-deps.sh
.venv/bin/python scripts/preflight.py --ids holes/math-ct-full.ids.txt
```

The setup script checks the live formula → babashka → LaTeXML → `:structure`
chain. Preflight also checks the remaining run inputs and model endpoint;
configure those as described in the mark7 handoff. A missing or broken LaTeXML
installation must be fixed before the run.
