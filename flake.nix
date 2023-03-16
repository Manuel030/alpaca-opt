{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system: with nixpkgs.legacyPackages.${system};
    {
      devShell = pkgs.mkShell {
        nativeBuildInputs = [ pkgs.bashInteractive ];
        buildInputs =
          let
            python-env = (python310.withPackages (p: with p; [
              pip
            ]));
          in
          [
            python-env
          ];
            LD_LIBRARY_PATH = lib.makeLibraryPath [
              stdenv.cc.cc
              zlib
          ];

      };
    });
}