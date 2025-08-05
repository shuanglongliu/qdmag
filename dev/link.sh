function link_all() {
  ln -sf ../core/*.py .
  ln -sf ../tools/tool*.py .
}

function link_yaml() {
  # ln -sf yaml_files/input_mn3.yaml input.yaml
  ln -sf yaml_files/input_mn3_isoJ.yaml input.yaml
  # ln -sf yaml_files/input_mn3_default_basis.yaml input.yaml
  # ln -sf yaml_files/input_hopzdo.yaml input.yaml
}

# link_all
link_yaml

