rm results.bbox.json
mv results.segm.json answer.json
zip $1 answer.json
rm answer.json
file=$1'.zip'
