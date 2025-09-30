# This script is executed during the setup process to initialize external dependencies.
import os
import subprocess


def setup_external():
    external_dir = "external"
    target_dir = os.path.join(external_dir, "wikitext")

    if not os.path.exists(".gitmodules"):
        print("Error: .gitmodules file is missing. Ensure the repository includes submodule configuration.")
        return

    if not os.path.exists(target_dir):
        os.makedirs(external_dir, exist_ok=True)
        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                check=True,
            )
            print(f"Initialized submodule at {target_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error initializing submodule: {e}")
    else:
        print(f"Submodule already exists at {target_dir}")


if __name__ == "__main__":
    setup_external()
