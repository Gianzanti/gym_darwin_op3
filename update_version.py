import subprocess

import toml


def get_git_version():
    """Gets the latest tag or commit hash from git."""
    try:
        # Try to get the latest tag
        tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], text=True).strip()
        # If no tag, try to get the short commit hash
        if not tag:
           tag = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return tag
    except subprocess.CalledProcessError:
        return "0.0.0"  # Default if no git history

def update_toml_version(version):
    """Updates the version in pyproject.toml."""
    try:
        with open("pyproject.toml", "r") as f:
            data = toml.load(f)

        # Find the correct section and key.  Handles different pyproject.toml structures
        if "tool" in data and "poetry" in data["tool"] and "version" in data["tool"]["poetry"]:
            data["tool"]["poetry"]["version"] = version
        elif "project" in data and "version" in data["project"]:
            data["project"]["version"] = version
        else:
            print("Warning: Could not find version key in pyproject.toml")
            return  # Don't modify the file if the key isn't found

        with open("pyproject.toml", "w") as f:
            toml.dump(data, f)

        print(f"Updated pyproject.toml version to: {version}")

    except FileNotFoundError:
        print("Error: pyproject.toml not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    version = get_git_version()
    print(f"Updating version to: {version}")
    update_toml_version(version)