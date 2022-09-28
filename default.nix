with import <nixpkgs> {};
with pkgs.python3Packages;

buildPythonPackage rec {
  pname = "energymodel";
  version = "0.2.0";

  propagatedBuildInputs = [
    tqdm
    matplotlib
    tensorflow
    tensorflow-datasets
  ];

  src = ./.;
}
