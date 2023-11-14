#!/bin/bash
set -e

function usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -t,  --tag <value>                 Tag to add to image
                                             The tag is set in the component specifications"
  echo "  -d,  --components-dir <value>      Directory containing components to build as subdirectories.
                                             The path should be relative to the root directory (default:src/components)"
  echo "  -r,  --registry <value>            The docker registry prefix to use (default: null for DockerHub)"
  echo "  -n,  --namespace <value>           The DockerHub namespace for the built images (default: ml6team)"
  echo "  -co, --component <value>           Specific component to build. Pass the component subdirectory name(s) to build
                                             certain component(s) or 'all' to build all components in the components
                                             directory (default: all)"
  echo "  -r,  --repo <value>                Set the repo (default: ml6team/fondant-usecase-RAG)"
  echo "  -l,  --label <value>               Set a container label, repeatable
                                             (e.g. org.opencontainers.image.source=https://github.com/ml6team/fondant-usecase-RAG)"
  echo "  -h,  --help                        Display this help message"
}

# Parse the arguments
while [[ "$#" -gt 0 ]]; do case $1 in
  -r |--registry) registry="$2"; shift;;
  -n |--namespace) namespace="$2"; shift;;
  -d |--components-dir ) components_dir="$2"; shift;;
  -r |--repo) repo="$2"; shift;;
  -t |--tag) tag=("$2"); shift;;
  -co|--component) components+=("$2"); shift;;
  -h |--help) usage; exit;;
  -l |--label) labels+=("$2"); shift;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

# Set default values for optional arguments if not passed
components_dir="${components_dir:-src/components}"
namespace="${namespace:-ml6team}"

# Get the component directory
scripts_dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
root_dir=$(dirname "$scripts_dir")
components_dir=$root_dir/$components_dir

# Determine the components to build
# Only directories that contains a Dockerfile will be considered for the component build
for dir in "$components_dir"/*/; do
  # Check if a Dockerfile exists in the current subdirectory
  if [ -f "$dir/Dockerfile" ]; then
      components_to_build+=("$dir")
  fi
done

# Loop through all subdirectories
for dir in "${components_to_build[@]}"; do
  pushd "$dir"
  BASENAME=${dir%/}
  BASENAME=${BASENAME##*/}

  full_image_name=${registry}/${namespace}/${BASENAME}:${tag}
  echo "Tagging image as $full_image_name"
  fondant build $dir -t $full_image_name --nocache "${labels[@]/#/--label }"

  popd
done
