docker run \
	--runtime=nvidia \
	-it \
	--rm \
	--ipc=host \
	-e LOCAL_USER_ID=`id -u $USER` \
	--mount src="$(pwd)",dst=/home/samsepiol/glide_finetune,type=bind \
	--mount src=/home/samsepiol/datasets/current-dataset,dst=/home/samsepiol/datasets/current_dataset,type=bind \
	'pyt_child'
