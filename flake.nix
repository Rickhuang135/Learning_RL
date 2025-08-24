{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.systems.url = "github:nix-systems/default";
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.systems.follows = "systems";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          system = "x86_64-linux";
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
        pythonEnv = pkgs.python313.withPackages (ps:
          with ps; [
          ]);
      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            uv
            glib
            zlib
            libGL
            stdenv.cc.cc.lib
            libsForQt5.wrapQtAppsHook
            ninja
            python313Packages.pandas
            (python313Packages.matplotlib.override {
              enableQt = true;
              enableGtk3 = true;
            })
          ];
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.libGL
            pkgs.glib.out
          ]}:$LD_LIBRARY_PATH";

          # UV_PYTHON = "${pythonEnv}/bin/python";
          UV_PYTHON_PREFERENCE = "only-system";
          shellHook = ''
            export PYTHONWARNINGS="ignore"
          '';
        };
      }
    );
}
