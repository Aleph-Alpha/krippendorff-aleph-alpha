v=$(uvx --from=toml-cli toml get --toml-path=pyproject.toml project.version)

# bump patch version
part="patch"
uvx --from bump2version bumpversion --allow-dirty --current-version "$v" "$part" pyproject.toml

v_new=$(uvx --from=toml-cli toml get --toml-path=pyproject.toml project.version)

git checkout -b auto-bump-version-${v_new}
git add pyproject.toml
git commit -m "chore(ci): bump version to $v_new"
git tag -a "$v_new" -m "$v_new"
git push --tags
git push --set-upstream origin auto-bump-version-${v_new} -o merge_request.create -o merge_request.merge_when_pipeline_succeeds -o merge_request.remove_source_branch   

