#!/bin/bash

# Get Dropbox shared link and local path from command-line arguments
local_path="data"
download_link="https://www.dropbox.com/scl/fo/p3fzn032ibrm83t3l2i7d/AKztDkDyeJ3aYqokJsRjZxQ?rlkey=llb2auihdx7xt5qy5sfkvfxtv&st=id2i1prz&dl=1"
mkdir $local_path

# Download the folder as a zip file
echo "Downloading Dropbox folder as zip..."
cd $local_path
if curl -L --fail --silent --show-error $download_link -o temp.zip; then
    echo "Download complete. Extracting files..."
    
    # Extract the zip file
    if unzip -q temp.zip; then
        echo "Folder contents extracted successfully to $local_path"
    else
        echo "Error: Failed to extract the zip file?"

    fi
else
    echo "Error: Download failed"
    echo "Curl exit code: $?"
    echo "Please check your internet connection and the Dropbox link"
    rm temp.zip
    exit 1
fi

# Clean up the temporary zip file
rm temp.zip

echo "Folder download and extraction complete!"



