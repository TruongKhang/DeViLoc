export dataset=data/Extended-CMU-Seasons
wget -r -np -nH -R "index.html*" --cut-dirs=4  https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/ -P $dataset
for slice in $dataset/*.tar
do
  tar -xf $slice -C $dataset && rm $slice
done