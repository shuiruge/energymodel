# This nix-shell file will build up an isolated Python environment that installs the modules in the requirements.txt, setting up TensorFlow correctly.
# C.f. https://discourse.nixos.org/t/pip-oserror-errno-30-read-only-file-system/16263/5
# and also https://nixos.wiki/wiki/Tensorflow#pip_install_in_a_nix-shell

with import <nixpkgs> {};
mkShell {
  name = "energymodel-shell";
  buildInputs = with python37.pkgs; [
    virtualenv
    pip
    setuptools
  ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages_10_1.cudatoolkit}/lib:${pkgs.cudaPackages_10_1.cudatoolkit}/lib:${pkgs.cudaPackages_10_1.cudatoolkit.lib}/lib:$LD_LIBRARY_PATH
    alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' TMPDIR='$HOME' \pip"
    export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.7/site-packages:$PYTHONPATH"
    export PATH="$(pwd)/_build/pip_packages/bin:$PATH"
    unset SOURCE_DATE_EPOCH
    virtualenv venv
    source venv/bin/activate
    pip install -e . -r requirements.txt
  '';
}
