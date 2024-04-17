export dataset=data/aachen
wget -r -np -nH -R "index.html*,aachen_v1_1.zip" --cut-dirs=4  https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/ -P $dataset
unzip $dataset/images/database_and_query_images.zip -d $dataset/images/
export fileid=1w_YRMsQfOo24asVFOU3PGWmb3-ncd-4E
export filename=$dataset/3D-models/aachen_v_1_0.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$fileid" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$fileid" -O $filename && rm -rf /tmp/cookies.txt
unzip $filename -d $dataset/3D-models/