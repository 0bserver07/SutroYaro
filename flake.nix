{
  description = "SutroYaro - Sparse Parity Energy Efficiency Research";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            buildInputs = [
              # Python with core deps
              (pkgs.python3.withPackages (p: [
                p.numpy
                # Docs site
                p.mkdocs-material
                p.mkdocs-mermaid2-plugin
                p.pymdown-extensions
              ]))
              # Telegram sync
              pkgs.bun
              # Google Docs sync
              pkgs.pandoc
            ];
            shellHook = ''
              export PYTHONPATH=$PWD/src:$PYTHONPATH
            '';
          };
        });
    };
}
