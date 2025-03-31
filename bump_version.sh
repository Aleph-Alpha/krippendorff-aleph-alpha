v=$(uvx --from=toml-cli toml get --toml-path=pyproject.toml project.version)

# bump patch version
part="patch"
uvx --from bump2version bumpversion --allow-dirty --current-version "$v" "$part" pyproject.toml

# git config user.name "ci-bot"
# git remote add gitlab_origin https://oauth2:$CI_JOB_TOKEN@gitlab.com/aleph-alpha/krippendorff-alpha.git
# git branch -f auto-bump-version
# git add pyproject.toml
# git commit -m "chore(ci): bump version to $v"
# git push --set-upstream origin auto-bump-version -o merge_request.create -o merge_request.merge_when_pipeline_succeeds

