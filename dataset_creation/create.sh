if [[ $# -eq 0 ]];
then echo " Need arguments for LABELS directory and IMAGES directory"
fi


# all the imagefile prefixes and labels are stored in a textfile :LABELS.txt
rm -rf LABELS.txt
labelsFile="./LABELS.txt"

# all the images from the extracted folder that actually have a valid label are filtered and put in folder :IMAGES
rmdir -rf IMAGES
mkdir IMAGES
imagesFolder='./IMAGES'
imagesFolder=$(readlink -m $imagesFolder)

#find all the leaf files in EMOTIONS folder .These leaf files contain the label of the images under its corresponding parent directory
files_containing_label=$(find $1 -type f)

#enter the imagefile name's prefix (i.e the folder name) as well as the label into LABELS.txt
for file in $files_containing_label
do
	echo "$file"
	label=$(printf "%d" `cat $file`)

	prefix=$(echo $file|xargs basename|cut -d _ -f 1,2)
	echo $prefix $label>>$labelsFile
done


imgdirs=$(find $2 -type d -mindepth 3 -maxdepth 3)

for dir in $imgdirs
do
	(
	subject_n_session=$(echo $dir| rev | cut -d / -f 1,2 |rev | sed 's/\//_/')
	if grep -q $subject_n_session $labelsFile
	then
		cd $dir
		ls -r|head -n 5| xargs cp -t $imagesFolder
	fi
	)
done
