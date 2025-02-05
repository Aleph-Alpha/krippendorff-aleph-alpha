from pathlib import Path
import semantic_version
import toml
from git import Repo


def create_release_tag() -> None:
    # Read version from pyproject.toml
    with open("pyproject.toml", "r") as f:
        config = toml.load(f)
        version = semantic_version.Version(config["project"]["version"])

    # Get the Git repository
    repo = Repo(Path())

    # Check if the version in pyproject.toml is greater than the latest tag
    if repo.tags:
        most_recent_tag = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)[-1]
        last_version = semantic_version.Version(most_recent_tag.name)
        if version <= last_version:
            raise ValueError("Version in pyproject.toml must be greater than the last release version")

    # Create the release tag
    tag_name = str(version)
    repo.create_tag(tag_name, message=f"Release {tag_name}")
    print(f"Created release tag {tag_name}")

    # Optionally, push the tag to the remote repository
    origin = repo.remote(name='origin')
    origin.push(tag_name)
    print(f"Pushed release tag {tag_name} to remote")


if __name__ == "__main__":
    create_release_tag()
