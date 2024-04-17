export dataset=data/cambridge
export scenes=( "KingsCollege" "OldHospital" "StMarysChurch" "ShopFacade" "GreatCourt" )
export IDs=( "251342" "251340" "251294" "251336" "251291" )
for i in "${!scenes[@]}"
do
  wget https://www.repository.cam.ac.uk/bitstream/handle/1810/${IDs[i]}/${scenes[i]}.zip -P $dataset
  unzip $dataset/${scenes[i]}.zip -d $dataset && rm $dataset/${scenes[i]}.zip
done

export fileid=1esqzZ1zEQlzZVic-H32V6kkZvc4NeS15
export filename=$dataset/CambridgeLandmarks_Colmap_Retriangulated_1024px.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$fileid" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$fileid" -O $filename && rm -rf /tmp/cookies.txt
unzip $filename -d $dataset