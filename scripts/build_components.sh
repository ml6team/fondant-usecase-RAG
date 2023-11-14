#!/bin/bash
set -e

function usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -t,  --tag <value>                 Tag to add to image, repeatable
                                             The first tag is set in the component specifications"
  echo "  -d,  --components-dir <value>      Directory containing components to build as subdirectories.
                                             The path should be relative to the root directory (default:components)"
  echo "  -r,  --registry <value>            The docker registry prefix to use (default: null for DockerHub)"
  echo "  -n,  --namespace <value>           The DockerHub namespace for the built images (default: fndnt)"
  echo "  -co, --component <value>           Specific component to build. Pass the component subdirectory name(s) to build
                                             certain component(s) or 'all' to build all components in the components
                                             directory (default: all)"
  echo "  -r,  --repo <value>                Set the repo (default: ml6team/fondant-usecase-RAG)"
  echo "  -l,  --label <value>               Set a container label (e.g. org.opencontainers.image.source=https://github.com/ml6team/fondant-usecase-RAG)"
  echo "  -h,  --help                        Display this help message"
}

# Parse the arguments
while [[ "$#" -gt 0 ]]; do case $1 in
  -r |--registry) registry="$2"; shift;;
  -n |--namespace) namespace="$2"; shift;;
  -d |--components-dir ) components_dir="$2"; shift;;
  -r |--repo) repo="$2"; shift;;
  -t |--tag) tags+=("$2"); shift;;
  -co|--component) components+=("$2"); shift;;
  -h |--help) usage; exit;;
  -l |--label) labels+=("$2"); shift;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

# Check for required argument
if [ -z "${tags}" ]; then
  echo "Error: tag parameter is required"
  usage
  exit 1
fi

# Set default values for optional arguments if not passed
component="${components:-all}"
components_dir="${components_dir:-components}"
namespace="${namespace:-ml6team}"
repo="${repo:-ml6team/fondant-usecase-RAG}"

# Get the component directory
scripts_dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
root_dir=$(dirname "$scripts_dir")
components_dir=$root_dir/"src"/${components_dir}

# Determine the components to build
if [[ "${component}" == "all" ]]; then
  for dir in "$components_dir"/*/; do
    # Check if a Dockerfile exists in the current subdirectory
    if [ -f "$dir/Dockerfile" ]; then
        components_to_build+=("$dir")
    fi
  done
else
  for component in "${components[@]}"; do
    components_to_build+=("$components_dir/${component}/")
  done
fi

# Loop through all subdirectories
for dir in "${components_to_build[@]}"; do
  pushd "$dir"
  BASENAME=${dir%/}
  BASENAME=${BASENAME##*/}

  full_image_names=()
  echo "Tagging image with following tags:"
  for tag in "${tags[@]}"; do
    full_image_name=${registry}/${namespace}/${BASENAME}:${tag}
    echo "$full_image_name"
    full_image_names+=("$full_image_name")
  done

  args=()

  exit

  # Add argument for each tag
  for tag in "${full_image_names[@]}" ; do
    args+=(-t "$tag")
  done

  echo "Freezing Fondant dependency version to ${tags[0]}"

  # TODO: fondant build label?
  docker build --push "${args[@]}" \
   --label org.opencontainers.image.source=https://github.com/${namespace}/${repo} \
   .

  popd

done
