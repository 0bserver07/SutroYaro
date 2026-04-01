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

              # Load Telegram credentials from sops-nix secrets
              for var in telegram_api_id telegram_api_hash telegram_bot_token sutro_group_chat_id; do
                if [ -f "/run/secrets/$var" ]; then
                  export "$(echo "$var" | tr '[:lower:]' '[:upper:]')"="$(cat "/run/secrets/$var")"
                fi
              done
            '';
          };
        });
    };
}
