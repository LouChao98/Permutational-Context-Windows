#! /bin/bash

# cli

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
  -f | --full)
    FULL=YES
    shift # past argument
    ;;
  *) # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift              # past argument
    ;;
  esac
done

# path
folder=${PWD##*/}
declare -A targets=(["group1"]="${folder}" ["group2"]="${folder}" ["ai2"]="project/${folder}")

if [ ${#POSITIONAL[@]} -eq 0 ]; then
  echo "No target provided. Sync all."
  POSITIONAL=("${!targets[@]}")
fi

for target in "${POSITIONAL[@]}"; do
  echo "==============="
  echo "syncing ${target}"
  if [ -z "${targets[$target]}" ]; then
    echo "Unrecognized ${target}. Skipping."
    continue
  fi
  rsync -avhHLP "$PWD"/ "$target:${targets[$target]}" --exclude-from=rsync-exclude #--delete

  # if [ -z "$(git status --porcelain)" ]; then
  #   # if clean, update .git, so wandb could trace versions
  #   # shellcheck disable=SC2029
  #   ssh "$target" "cd ${targets[$target]}; git fetch --all; git reset --hard origin/main"
  # fi

  if [ "${FULL}" = YES ]; then
    rsync -avhHLP "$PWD"/data "$target:${targets[$target]}" --exclude={.idea,.git,__pycache__,cache}
    # rsync -avhHLP "$PWD"/logs/saved "${targets[$target]}/logs" --exclude={.idea,.git,__pycache__,cache}
  fi
done
