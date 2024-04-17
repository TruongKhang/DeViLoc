export dataset=data/RobotCarSeasons
wget -r -np -nH -R "index.html*" --cut-dirs=4  https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/ -P $dataset
for condition in $dataset/images/*.zip
do
  unzip condition -d $dataset/images/
done