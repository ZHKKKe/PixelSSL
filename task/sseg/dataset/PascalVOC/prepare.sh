
echo "Start to prepare PascalVOC 2012 with SBD...\n"

pascalvoc_checksum=e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb
pascalvoc_tar_name=VOCtrainval_11-May-2012.tar
pascalvoc_folder=VOCdevkit

sbd_checksum=63b2c2e40badf93e7c4a91e2c5e6dd2eb68ace6a639736f9a2447b446ec2a13d
sbd_zip_name=SegmentationClassAug.zip
sbd_folder=SegmentationClassAug

# download PascalVOC 2012
if [ ! -f "$pascalvoc_tar_name" ]; then
    echo "Downloading PascalVOC 2012..."
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
fi
echo "PascalVOC 2012 is downloaded.\n"

# check the integrity of PascalVOC 2012
if ! echo "$pascalvoc_checksum $pascalvoc_tar_name" | sha256sum -c; then
    echo "[ERROR] SHA256SUM of the file $pascalvoc_tar_name is invalid. Please delete and re-download it.\n"
    exit 1
fi

# extract PascalVOC 2012
if [ -d "$pascalvoc_folder" ]; then
    rm -r "$pascalvoc_folder"
fi
echo "Unzip PascalVOC 2012..."
tar -xvf "$pascalvoc_tar_name"
echo "PascalVOC 2012 is extracted.\n"

# download SBD
if [ ! -f "$sbd_zip_name" ]; then
    echo "Downloading the augmented labels (Segmentation Boundaries Dataset [SBD])..."
    wget http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip
fi
echo "SBD is downloaded.\n"

# check the integrity of SBD
if ! echo "$sbd_checksum $sbd_zip_name" | sha256sum -c; then
    echo "[ERROR] SHA256SUM of the file $sbd_zip_name is invalid. Please delete and re-download it.\n"
    exit 1
fi

# extract SBD
if [ -d "$sbd_folder" ]; then
    rm -r "$sbd_folder"
fi
echo "Unzip SBD..."
unzip SegmentationClassAug.zip
echo "SBD is extracted.\n"

# merge two datasets
echo "Merge SBD to PascalVOC 2012..."
mv "$sbd_folder" "$pascalvoc_folder/VOC2012"
echo "SBD is merged to PascalVOC 2012.\n"
rm -r "__MACOSX"

# create the samples list file
echo "Create the samples list for PascalVOC 2012 with SBD ['train_aug.txt']..."
python tool/list_augtrain_samples.py
echo "The samples list for PascalVOC 2012 with SBD ['train_aug,txt'] is created.\n"

echo "PascalVOC 2012 augmented by SBD is available in the folder 'VOCdevkit'"
echo "The augmented samples list file is 'VOCdevkit/VOC2012/ImageSets/Segmentation/train_aug.txt'"
