# In case the command line have already been installed, "true" will make it return a non-zero error code to avoid stopping the scripts
xcode-select --install || true
bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
brew update
