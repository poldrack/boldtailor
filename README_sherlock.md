apptainer exec --cleanenv \
  --env UV_PROJECT_ENVIRONMENT=.venv-apptainer,UV_PYTHON_DOWNLOADS=never \
  $GROUP_HOME/containers/uv-python3.12-bookworm.sif \
  uv sync --link-mode hardlink --python 3.12
